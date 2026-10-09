"""Anomaly detection and deterministic explainable classification service for Twinergy 2.0.

Provides:
1. Primary Residual-Based Energy Anomaly Analysis & Classification (EnergyAnomalyAnalyzer):
   Evaluates actual energy consumption against expected energy consumption (predicted by
   EnergyForecaster or provided baseline), computing residual and normalized deviation metrics,
   and classifying anomalies into 5 deterministic types (NORMAL, HIGH_DEMAND, LOW_DEMAND,
   RAPID_SPIKE, PERSISTENT_HIGH) with human-readable classification reasons.
2. Auxiliary Isolation Forest Signal (AnomalyDetector):
   Wraps the trained Isolation Forest artifact (twinergy_isoforest.pkl) as an auxiliary signal.

───────────────────────────────────────────────────────────────────────────────
DETERMINISTIC ANOMALY CLASSIFICATION RULES & PRIORITY ORDER:
1. Priority 1: NORMAL
   - Condition: abs_deviation_ratio <= 0.25 (severity == NORMAL)
   - Reason: Consumption is within normal baseline range.

2. Priority 2: LOW_DEMAND
   - Condition: severity != NORMAL and residual < 0
   - Reason: Actual consumption is significantly below expected baseline.

3. Priority 3: RAPID_SPIKE
   - Condition: severity != NORMAL, residual > 0, actual_kwh > 1.35 * rolling_mean_6hr_kwh,
                and deviation_ratio > 0.35
   - Reason: Sudden, sharp consumption surge relative to 6-hour rolling baseline.

4. Priority 4: PERSISTENT_HIGH
   - Condition: severity != NORMAL, residual > 0, and (consecutive_anomaly_streak >= 3
                or rolling_mean_24hr_kwh > 1.25 * expected_kwh)
   - Reason: Sustained elevated consumption across consecutive hourly readings.

5. Priority 5: HIGH_DEMAND
   - Condition: severity != NORMAL and residual > 0 (fallback positive anomaly)
   - Reason: Actual consumption exceeded expected baseline significantly.

───────────────────────────────────────────────────────────────────────────────
CRITICAL DOMAIN & CAUSALITY WARNINGS:
1. NO SUPERVISED ML CLASSIFIER / GROUND TRUTH LABELS:
   Classification rules are purely deterministic and rule-based. No ground-truth labels exist.

2. NO CAUSAL PROOF:
   Classification identifies statistical divergence patterns (e.g. RAPID_SPIKE vs PERSISTENT_HIGH).
   It does NOT prove physical root causes (e.g. equipment fault vs occupancy shift).
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loader import get_default_project_root
from src.data.schema import ANOMALY_FEATURES, LIGHTGBM_FEATURES

BUILDING_20_ISOFOREST_WARNING = (
    "Artifact twinergy_isoforest.pkl was trained exclusively on Building 20. "
    "Multi-building cross-inference is uncalibrated and preserved solely as auxiliary supporting evidence."
)

CAUSALITY_WARNING = (
    "Residual deviation indicates statistical divergence from expected baseline, "
    "not physical causal proof."
)


def resolve_isoforest_model_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve Isolation Forest model file path with precedence."""
    if custom_path is not None:
        p = Path(custom_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured Isolation Forest path does not exist: {p}")
        return p

    env_path = os.getenv("TWINERGY_ISOFOREST_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"TWINERGY_ISOFOREST_PATH does not exist: {p}")
        return p

    root = get_default_project_root()
    candidates = [
        root / "twinergy_isoforest.pkl",
        root / "models" / "twinergy_isoforest.pkl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"No Isolation Forest model artifact found under {root}. "
        "Please provide a path or set TWINERGY_ISOFOREST_PATH."
    )


class EnergyAnomalyAnalyzer:
    """Primary service for evaluating residual-based energy consumption anomalies and deterministic classification."""

    EPSILON: float = 1e-3  # 0.001 kWh minimum denominator for zero/near-zero expected handling
    WARNING_THRESHOLD: float = 0.25  # 25% deviation ratio
    CRITICAL_THRESHOLD: float = 0.50  # 50% deviation ratio

    def __init__(self, forecaster: Optional[Any] = None):
        """Initialize EnergyAnomalyAnalyzer.

        Args:
            forecaster: Optional EnergyForecaster instance to compute expected consumption.
        """
        self.forecaster = forecaster

    def classify_point(
        self,
        actual_kwh: float,
        expected_kwh: float,
        rolling_mean_6hr_kwh: Optional[float] = None,
        rolling_mean_24hr_kwh: Optional[float] = None,
        streak_hours: int = 1,
        epsilon: float = EPSILON,
    ) -> Dict[str, Any]:
        """Classify an anomaly point into a deterministic anomaly_type and reason.

        Priority Order:
        1. NORMAL: if severity == NORMAL (abs_dev <= 0.25)
        2. LOW_DEMAND: if residual < 0 and severity != NORMAL
        3. RAPID_SPIKE: if residual > 0, actual_kwh > 1.35 * rolling_6hr, and dev_ratio > 0.35
        4. PERSISTENT_HIGH: if residual > 0 and (streak_hours >= 3 or rolling_24hr > 1.25 * expected)
        5. HIGH_DEMAND: if residual > 0 and severity != NORMAL (fallback positive anomaly)

        Returns:
            Dict with anomaly_type, classification_reason, and priority_rank.
        """
        actual = max(0.0, float(actual_kwh))
        expected = max(0.0, float(expected_kwh))
        residual = actual - expected
        effective_exp = max(expected, epsilon)
        dev_ratio = residual / effective_exp
        abs_dev = abs(dev_ratio)

        # Determine severity
        if abs_dev > self.CRITICAL_THRESHOLD:
            severity = "CRITICAL"
        elif abs_dev > self.WARNING_THRESHOLD:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        is_anomaly = 1 if severity in ("WARNING", "CRITICAL") else 0

        # Priority 1: NORMAL
        if is_anomaly == 0:
            return {
                "anomaly_type": "NORMAL",
                "classification_reason": (
                    f"Consumption is within normal range (relative deviation: {dev_ratio * 100.0:+.1f}%)."
                ),
                "priority_rank": 1,
                "severity": severity,
                "is_anomaly": 0,
            }

        # Priority 2: LOW_DEMAND
        if residual < 0:
            return {
                "anomaly_type": "LOW_DEMAND",
                "classification_reason": (
                    f"Low demand detected: actual consumption ({actual:.1f} kWh) is "
                    f"{abs(dev_ratio) * 100.0:.1f}% below expected baseline ({expected:.1f} kWh)."
                ),
                "priority_rank": 2,
                "severity": severity,
                "is_anomaly": 1,
            }

        # Positive anomaly checks (residual > 0)

        # Priority 3: RAPID_SPIKE
        if rolling_mean_6hr_kwh is not None and rolling_mean_6hr_kwh > 0:
            r6_kwh = float(rolling_mean_6hr_kwh)
            surge_ratio = (actual - r6_kwh) / max(r6_kwh, epsilon)
            if surge_ratio > 0.35 and dev_ratio > 0.35:
                return {
                    "anomaly_type": "RAPID_SPIKE",
                    "classification_reason": (
                        f"Rapid consumption spike detected: actual consumption ({actual:.1f} kWh) is "
                        f"{surge_ratio * 100.0:.1f}% above the 6-hour rolling baseline ({r6_kwh:.1f} kWh) "
                        f"and {dev_ratio * 100.0:.1f}% above expected."
                    ),
                    "priority_rank": 3,
                    "severity": severity,
                    "is_anomaly": 1,
                }

        # Priority 4: PERSISTENT_HIGH
        if streak_hours >= 3:
            return {
                "anomaly_type": "PERSISTENT_HIGH",
                "classification_reason": (
                    f"Persistent high demand detected: elevated consumption has sustained across "
                    f"{streak_hours} consecutive hourly readings (actual: {actual:.1f} kWh, expected: {expected:.1f} kWh)."
                ),
                "priority_rank": 4,
                "severity": severity,
                "is_anomaly": 1,
            }
        elif rolling_mean_24hr_kwh is not None and rolling_mean_24hr_kwh > 0:
            r24_kwh = float(rolling_mean_24hr_kwh)
            if r24_kwh > 1.25 * expected and dev_ratio > 0.25:
                return {
                    "anomaly_type": "PERSISTENT_HIGH",
                    "classification_reason": (
                        f"Persistent high demand detected: 24-hour mean consumption ({r24_kwh:.1f} kWh) is "
                        f"{((r24_kwh - expected) / expected) * 100.0:.1f}% above baseline and actual "
                        f"({actual:.1f} kWh) is {dev_ratio * 100.0:.1f}% above expected."
                    ),
                    "priority_rank": 4,
                    "severity": severity,
                    "is_anomaly": 1,
                }

        # Priority 5: HIGH_DEMAND
        return {
            "anomaly_type": "HIGH_DEMAND",
            "classification_reason": (
                f"High demand detected: actual consumption ({actual:.1f} kWh) exceeded expected baseline "
                f"({expected:.1f} kWh) by {dev_ratio * 100.0:.1f}%."
            ),
            "priority_rank": 5,
            "severity": severity,
            "is_anomaly": 1,
        }

    def analyze_point(
        self,
        actual_kwh: float,
        expected_kwh: float,
        epsilon: float = EPSILON,
    ) -> Dict[str, Any]:
        """Analyze a single observation of actual vs expected consumption."""
        res = self.classify_point(actual_kwh=actual_kwh, expected_kwh=expected_kwh, epsilon=epsilon)
        actual = max(0.0, float(actual_kwh))
        expected = max(0.0, float(expected_kwh))
        residual = actual - expected
        effective_expected = max(expected, epsilon)
        dev_ratio = residual / effective_expected

        return {
            "actual_kwh": actual,
            "expected_kwh": expected,
            "residual": residual,
            "deviation_ratio": dev_ratio,
            "abs_deviation_ratio": abs(dev_ratio),
            "severity": res["severity"],
            "is_anomaly": res["is_anomaly"],
            "anomaly_direction": "HIGH_DEMAND" if residual > 0 and res["is_anomaly"] else ("LOW_DEMAND" if residual < 0 and res["is_anomaly"] else "NOMINAL"),
            "anomaly_type": res["anomaly_type"],
            "classification_reason": res["classification_reason"],
        }

    def analyze(
        self,
        df: pd.DataFrame,
        expected_kwh: Optional[Union[Sequence[float], np.ndarray, str]] = None,
        actual_kwh: Optional[Union[Sequence[float], np.ndarray, str]] = None,
        epsilon: float = EPSILON,
    ) -> pd.DataFrame:
        """Analyze DataFrame of energy consumption records for residual anomalies and deterministic classification.

        Args:
            df: Input DataFrame.
            expected_kwh: Array/series of expected values, column name in df, or None
                          to predict via forecaster or compute fallback baseline.
            actual_kwh: Array/series of actual values, column name in df, or None
                        to infer from meter_reading / meter_reading_log.
            epsilon: Minimum denominator for zero expected handling.

        Returns:
            Enriched DataFrame with residual anomaly metrics and classification columns.
        """
        out = df.copy()

        # 1. Resolve actual_kwh
        if actual_kwh is not None:
            if isinstance(actual_kwh, str) and actual_kwh in out.columns:
                act_vals = out[actual_kwh].values.astype(np.float64)
            else:
                act_vals = np.asarray(actual_kwh, dtype=np.float64)
        elif "actual" in out.columns:
            act_vals = out["actual"].values.astype(np.float64)
        elif "actual_kwh" in out.columns:
            act_vals = out["actual_kwh"].values.astype(np.float64)
        elif "meter_reading" in out.columns:
            act_vals = out["meter_reading"].values.astype(np.float64)
        elif "meter_reading_log" in out.columns:
            act_vals = np.expm1(out["meter_reading_log"].values.astype(np.float64))
        else:
            raise ValueError(
                "Could not resolve actual consumption column. "
                "Must provide 'actual_kwh', 'meter_reading', or 'meter_reading_log'."
            )

        act_vals = np.clip(act_vals, 0.0, None)

        # 2. Resolve expected_kwh
        if expected_kwh is not None:
            if isinstance(expected_kwh, str) and expected_kwh in out.columns:
                exp_vals = out[expected_kwh].values.astype(np.float64)
            else:
                exp_vals = np.asarray(expected_kwh, dtype=np.float64)
        elif "expected" in out.columns:
            exp_vals = out["expected"].values.astype(np.float64)
        elif "expected_kwh" in out.columns:
            exp_vals = out["expected_kwh"].values.astype(np.float64)
        elif "predicted" in out.columns:
            exp_vals = out["predicted"].values.astype(np.float64)
        elif "predicted_kwh" in out.columns:
            exp_vals = out["predicted_kwh"].values.astype(np.float64)
        elif (
            self.forecaster is not None
            and all(col in out.columns for col in LIGHTGBM_FEATURES)
            and not out[LIGHTGBM_FEATURES].isnull().any().any()
        ):
            exp_vals = self.forecaster.predict(out[LIGHTGBM_FEATURES], return_kwh=True)
        elif "rolling_mean_6hr" in out.columns:
            exp_vals = np.where(
                out["rolling_mean_6hr"] > 0,
                np.expm1(out["rolling_mean_6hr"]),
                act_vals,
            )
        else:
            exp_vals = act_vals.copy()

        exp_vals = np.clip(exp_vals, 0.0, None)

        # 3. Compute residual metrics
        residual = act_vals - exp_vals
        effective_expected = np.maximum(exp_vals, epsilon)
        dev_ratio = residual / effective_expected
        abs_dev_ratio = np.abs(dev_ratio)

        severity = np.where(
            abs_dev_ratio > self.CRITICAL_THRESHOLD,
            "CRITICAL",
            np.where(abs_dev_ratio > self.WARNING_THRESHOLD, "WARNING", "NORMAL"),
        )

        is_anomaly = np.where((severity == "CRITICAL") | (severity == "WARNING"), 1, 0)

        direction = np.where(
            is_anomaly == 1,
            np.where(residual > 0, "HIGH_DEMAND", "LOW_DEMAND"),
            "NOMINAL",
        )

        # 4. Resolve rolling Baselines
        r6_kwh = (
            np.expm1(out["rolling_mean_6hr"].values)
            if "rolling_mean_6hr" in out.columns
            else None
        )
        r24_kwh = (
            np.expm1(out["rolling_mean_24hr"].values)
            if "rolling_mean_24hr" in out.columns
            else None
        )

        # Compute consecutive streak hours for positive anomalies
        streak = np.zeros(len(out), dtype=int)
        curr_streak = 0
        for i in range(len(out)):
            if is_anomaly[i] == 1 and residual[i] > 0:
                curr_streak += 1
            else:
                curr_streak = 0
            streak[i] = curr_streak

        # 5. Classify each row deterministically
        anomaly_types: List[str] = []
        reasons: List[str] = []

        for i in range(len(out)):
            r6_val = float(r6_kwh[i]) if r6_kwh is not None else None
            r24_val = float(r24_kwh[i]) if r24_kwh is not None else None
            c_res = self.classify_point(
                actual_kwh=act_vals[i],
                expected_kwh=exp_vals[i],
                rolling_mean_6hr_kwh=r6_val,
                rolling_mean_24hr_kwh=r24_val,
                streak_hours=int(streak[i]),
                epsilon=epsilon,
            )
            anomaly_types.append(c_res["anomaly_type"])
            reasons.append(c_res["classification_reason"])

        out["actual_kwh"] = act_vals
        out["expected_kwh"] = exp_vals
        out["residual"] = residual
        out["deviation_ratio"] = dev_ratio
        out["abs_deviation_ratio"] = abs_dev_ratio
        out["severity"] = severity
        out["is_anomaly"] = is_anomaly
        out["anomaly_direction"] = direction
        out["anomaly_type"] = anomaly_types
        out["classification_reason"] = reasons

        return out


class AnomalyDetector:
    """Service for anomaly detection combining primary residual analysis and auxiliary Isolation Forest."""

    TRAINING_SCOPE_NOTE = BUILDING_20_ISOFOREST_WARNING

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        forecaster: Optional[Any] = None,
    ):
        """Initialize AnomalyDetector with Isolation Forest model and optional Forecaster.

        Args:
            model_path: Optional path to twinergy_isoforest.pkl.
            forecaster: Optional EnergyForecaster for expected energy baseline.
        """
        self.model_path = resolve_isoforest_model_path(model_path)
        self.model = self._load_model(self.model_path)
        self.feature_names: List[str] = list(ANOMALY_FEATURES)
        self._verify_feature_count()
        self.analyzer = EnergyAnomalyAnalyzer(forecaster=forecaster)

    def _load_model(self, path: Path) -> Any:
        """Load pickled Isolation Forest model."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(path, "rb") as f:
                model = pickle.load(f)
        return model

    def _verify_feature_count(self) -> None:
        """Verify model expects 8 features."""
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features is not None and n_features != len(self.feature_names):
            raise ValueError(
                f"Model expected {n_features} features, but schema defines {len(self.feature_names)}"
            )

    def prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Validate and align feature matrix to 8-feature Isolation Forest schema."""
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input data is missing required anomaly features: {missing}")
            return X[self.feature_names].copy()
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected 2D array with {len(self.feature_names)} features, got shape {X.shape}"
                )
            return pd.DataFrame(X, columns=self.feature_names)
        else:
            raise TypeError(f"Unsupported feature type: {type(X)}. Expected pd.DataFrame or np.ndarray.")

    def predict_raw(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Raw Isolation Forest prediction: -1 for anomaly, 1 for nominal."""
        features_df = self.prepare_features(X)
        return self.model.predict(features_df)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Binary Isolation Forest prediction: 1 for anomaly, 0 for nominal."""
        raw = self.predict_raw(X)
        return (raw == -1).astype(int)

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute Isolation Forest anomaly score."""
        features_df = self.prepare_features(X)
        return self.model.score_samples(features_df)

    def detect(
        self,
        df: pd.DataFrame,
        expected_kwh: Optional[Union[Sequence[float], np.ndarray, str]] = None,
    ) -> pd.DataFrame:
        """Run residual-based anomaly detection primary engine with auxiliary Isolation Forest signals.

        Computes:
        - Primary Residual Signal: actual_kwh, expected_kwh, residual, deviation_ratio,
          abs_deviation_ratio, severity (NORMAL/WARNING/CRITICAL), is_anomaly, anomaly_direction,
          anomaly_type (NORMAL/HIGH_DEMAND/LOW_DEMAND/RAPID_SPIKE/PERSISTENT_HIGH), classification_reason.
        - Auxiliary Isolation Forest Signal: isoforest_anomaly, isoforest_score, isoforest_warning.

        Args:
            df: Input DataFrame.
            expected_kwh: Optional array or column name for expected kWh.

        Returns:
            Enriched DataFrame with anomaly columns.
        """
        # 1. Primary residual-based analysis & classification
        out = self.analyzer.analyze(df, expected_kwh=expected_kwh)

        # 2. Auxiliary Isolation Forest evaluation if 8 features are present
        has_iso_features = all(col in out.columns for col in self.feature_names)
        if has_iso_features:
            iso_df = self.prepare_features(out)
            iso_raw = self.model.predict(iso_df)
            out["isoforest_anomaly"] = (iso_raw == -1).astype(int)
            out["isoforest_score"] = self.model.score_samples(iso_df)
            out["isoforest_warning"] = BUILDING_20_ISOFOREST_WARNING
            out["anomaly_score"] = iso_raw
        else:
            out["isoforest_anomaly"] = 0
            out["isoforest_score"] = 0.0
            out["isoforest_warning"] = BUILDING_20_ISOFOREST_WARNING
            out["anomaly_score"] = 1

        return out

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return operational metadata and domain warning."""
        return {
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "num_features": len(self.feature_names),
            "features": list(self.feature_names),
            "training_scope": "Building 20 only (Auxiliary signal)",
            "warning": BUILDING_20_ISOFOREST_WARNING,
            "causality_note": CAUSALITY_WARNING,
            "n_estimators": getattr(self.model, "n_estimators", None),
            "contamination": getattr(self.model, "contamination", None),
        }


def load_anomaly_detector(
    model_path: Optional[Union[str, Path]] = None,
    forecaster: Optional[Any] = None,
) -> AnomalyDetector:
    """Convenience factory function to instantiate AnomalyDetector."""
    return AnomalyDetector(model_path=model_path, forecaster=forecaster)


def load_anomaly_analyzer(
    forecaster: Optional[Any] = None,
) -> EnergyAnomalyAnalyzer:
    """Convenience factory function to instantiate EnergyAnomalyAnalyzer."""
    return EnergyAnomalyAnalyzer(forecaster=forecaster)
