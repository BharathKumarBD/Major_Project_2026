"""Twinergy 2.0 Models Package.

Provides encapsulated services for:
- LightGBM energy forecasting (preserving 22-feature contract)
- Isolation Forest anomaly detection (preserving 8-feature contract with Building 20 domain documentation)
- Model explainability (global tree gain importance vs. local SHAP attribution)
"""

from src.models.anomaly import (
    AnomalyDetector,
    load_anomaly_detector,
    resolve_isoforest_model_path,
)
from src.models.explainability import (
    ModelExplainer,
    load_explainer,
)
from src.models.forecasting import (
    EnergyForecaster,
    load_forecaster,
    resolve_lgbm_model_path,
)

__all__ = [
    # Forecasting
    "EnergyForecaster",
    "load_forecaster",
    "resolve_lgbm_model_path",
    # Anomaly
    "AnomalyDetector",
    "load_anomaly_detector",
    "resolve_isoforest_model_path",
    # Explainability
    "ModelExplainer",
    "load_explainer",
]
