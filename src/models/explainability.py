"""Explainability service for Twinergy 2.0.

Provides:
1. Global tree-based feature importance from LightGBM (gain / split frequency).
2. Local, sample-bounded SHAP explanations using TreeExplainer.

CRITICAL DISTINCTION:
- LightGBM Gain Importance: Measures total reduction of training loss achieved by splits
  on each feature across all trees. It is NOT SHAP and must never be labeled as SHAP.
- SHAP (SHapley Additive exPlanations): Additive local attribution decomposing a specific
  prediction into feature contributions: f(x) = E[f(x)] + sum(phi_i).

GUARDRAIL:
SHAP calculation is deliberately bounded to small samples (default max 100 instances)
to prevent CPU/memory exhaustion on large time-series datasets. Full-dataset SHAP is
explicitly prevented.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.schema import LIGHTGBM_FEATURES
from src.models.forecasting import EnergyForecaster, resolve_lgbm_model_path

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelExplainer:
    """Service providing model explainability via global tree importance and local SHAP."""

    DEFAULT_MAX_SHAP_SAMPLES: int = 100

    def __init__(
        self,
        forecaster: Optional[Union[EnergyForecaster, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize ModelExplainer.

        Args:
            forecaster: An EnergyForecaster instance or raw LightGBM booster.
            model_path: Optional path to load EnergyForecaster if forecaster is None.
        """
        if isinstance(forecaster, EnergyForecaster):
            self.forecaster = forecaster
            self.model = forecaster.model
        elif forecaster is not None:
            self.model = forecaster
            self.forecaster = None
        else:
            self.forecaster = EnergyForecaster(model_path=model_path)
            self.model = self.forecaster.model

        self.feature_names: List[str] = list(LIGHTGBM_FEATURES)
        self._tree_explainer: Optional[Any] = None

    @property
    def tree_explainer(self) -> Any:
        """Lazily initialize shap.TreeExplainer."""
        if self._tree_explainer is None:
            if not HAS_SHAP:
                raise ImportError(
                    "The 'shap' package is required for SHAP explanations. "
                    "Install it via `pip install shap`."
                )
            self._tree_explainer = shap.TreeExplainer(self.model)
        return self._tree_explainer

    def get_tree_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Extract global tree-split feature importance from the LightGBM booster.

        NOTE: This is tree split gain/split count from LightGBM, NOT SHAP attribution.

        Args:
            importance_type: 'gain' (total split loss reduction) or 'split' (number of splits).

        Returns:
            pd.DataFrame with columns ['feature', 'importance', 'importance_type'],
            sorted descending by importance.
        """
        raw_imp = self.model.feature_importance(importance_type=importance_type)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": raw_imp,
            "importance_type": importance_type,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def _prepare_matrix(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Prepare and validate feature matrix matching the 22-feature contract."""
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])

        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input is missing required features for explainability: {missing}")
            df = X[self.feature_names].copy()
            if df["square_feet"].isnull().any():
                df["square_feet"] = df["square_feet"].fillna(df["square_feet"].median())
            return df
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected array with {len(self.feature_names)} features, got {X.shape}")
            return pd.DataFrame(X, columns=self.feature_names)
        else:
            raise TypeError(f"Unsupported data type for explainability: {type(X)}")

    def explain_instance(self, row: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Compute local additive SHAP explanation for a single prediction instance.

        Decomposes the prediction into:
            prediction_log = base_value + sum(shap_values)

        Args:
            row: Single observation (Series or 1-row DataFrame).

        Returns:
            Structured dictionary with base_value, predicted_log, predicted_kwh,
            and ordered feature attributions.
        """
        df = self._prepare_matrix(row)
        if len(df) != 1:
            df = df.iloc[[0]]

        explainer = self.tree_explainer
        shap_vals = explainer.shap_values(df)

        # Handle different SHAP output formats (array vs list of arrays)
        if isinstance(shap_vals, list):
            sv = shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray):
            sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
        else:
            sv = np.asarray(shap_vals).flatten()

        base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
        pred_log = float(base_val + np.sum(sv))
        pred_kwh = float(np.expm1(pred_log))

        feat_vals = df.iloc[0].to_dict()
        contributions = []
        for feat, val, phi in zip(self.feature_names, df.iloc[0], sv):
            contributions.append({
                "feature": feat,
                "value": float(val) if val is not None else None,
                "shap_value": float(phi),
                "abs_shap": float(abs(phi)),
            })

        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        return {
            "base_value": base_val,
            "predicted_log": pred_log,
            "predicted_kwh": pred_kwh,
            "contributions": contributions,
            "top_positive_drivers": [c for c in contributions if c["shap_value"] > 0][:5],
            "top_negative_drivers": [c for c in contributions if c["shap_value"] < 0][:5],
        }

    def explain_sample(
        self,
        df: pd.DataFrame,
        max_samples: int = DEFAULT_MAX_SHAP_SAMPLES,
    ) -> Dict[str, Any]:
        """Compute SHAP explanations for a bounded sample of observations.

        Safety guardrail: Caps sample size to `max_samples` to avoid runaway execution.

        Args:
            df: DataFrame containing the 22 features.
            max_samples: Maximum number of rows to evaluate (default 100).

        Returns:
            Dictionary with mean absolute SHAP values per feature, raw SHAP matrix,
            base_value, and evaluated sample size.
        """
        features_df = self._prepare_matrix(df)

        if len(features_df) > max_samples:
            warnings.warn(
                f"explain_sample received {len(features_df)} rows. "
                f"Sampling down to {max_samples} instances to prevent execution stalls.",
                UserWarning,
            )
            features_df = features_df.sample(max_samples, random_state=42)

        explainer = self.tree_explainer
        shap_vals = explainer.shap_values(features_df)

        if isinstance(shap_vals, list):
            sv = shap_vals[0]
        else:
            sv = np.asarray(shap_vals)

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        summary_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

        return {
            "sample_size": len(features_df),
            "base_value": base_val,
            "mean_abs_shap": summary_df,
            "shap_values": sv,
            "feature_names": list(self.feature_names),
        }


def load_explainer(
    forecaster: Optional[Union[EnergyForecaster, Any]] = None,
    model_path: Optional[str] = None,
) -> ModelExplainer:
    """Convenience factory function to instantiate ModelExplainer."""
    return ModelExplainer(forecaster=forecaster, model_path=model_path)
