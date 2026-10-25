"""Service container and helper utilities for Twinergy 2.0 API.

Manages singleton service instances, dependency injection providers,
and sanitization of internal filesystem paths from error responses.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd
from fastapi import HTTPException

from src.data.loader import DataLoader
from src.models.anomaly import (
    AnomalyDetector,
    EnergyAnomalyAnalyzer,
    load_anomaly_analyzer,
    load_anomaly_detector,
)
from src.models.explainability import ModelExplainer, load_explainer
from src.models.forecasting import EnergyForecaster, load_forecaster
from src.models.impact import EnergyImpactCalculator, load_impact_calculator
from src.models.recommendations import (
    RecommendationEngine,
    load_recommendation_engine,
)
from src.models.simulation import EnergySimulationEngine, load_simulation_engine
from src.models.twin_state import TwinStateBuilder

logger = logging.getLogger("twinergy.api")


def sanitize_error_message(message: str) -> str:
    """Sanitize internal filesystem paths and sensitive details from error messages."""
    if not message:
        return "An error occurred while processing the request."

    # Replace absolute system paths (e.g. /Users/... or /home/...) with generic token
    cleaned = re.sub(r"(/[a-zA-Z0-9_\.\-]+)+", "<dataset_path>", message)
    # Clean up file path references
    cleaned = re.sub(r"in <dataset_path>", "", cleaned).strip()
    return cleaned


def to_iso_string(ts: Any) -> Optional[str]:
    """Convert timestamp, datetime, or string to standard ISO 8601 string."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        parsed = pd.to_datetime(ts)
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return str(ts)


class ServiceContainer:
    """Singleton service container maintaining shared service references."""

    def __init__(self):
        self._loader: Optional[DataLoader] = None
        self._forecaster: Optional[EnergyForecaster] = None
        self._anomaly_analyzer: Optional[EnergyAnomalyAnalyzer] = None
        self._anomaly_detector: Optional[AnomalyDetector] = None
        self._impact_calculator: Optional[EnergyImpactCalculator] = None
        self._twin_state_builder: Optional[TwinStateBuilder] = None
        self._simulation_engine: Optional[EnergySimulationEngine] = None
        self._recommendation_engine: Optional[RecommendationEngine] = None

    def get_loader(self) -> DataLoader:
        if self._loader is None:
            self._loader = DataLoader()
        return self._loader

    def get_forecaster(self) -> EnergyForecaster:
        if self._forecaster is None:
            self._forecaster = load_forecaster()
        return self._forecaster

    def get_anomaly_analyzer(self) -> EnergyAnomalyAnalyzer:
        if self._anomaly_analyzer is None:
            self._anomaly_analyzer = load_anomaly_analyzer(forecaster=self.get_forecaster())
        return self._anomaly_analyzer

    def get_anomaly_detector(self) -> AnomalyDetector:
        if self._anomaly_detector is None:
            self._anomaly_detector = load_anomaly_detector(forecaster=self.get_forecaster())
        return self._anomaly_detector

    def get_impact_calculator(self) -> EnergyImpactCalculator:
        if self._impact_calculator is None:
            self._impact_calculator = load_impact_calculator()
        return self._impact_calculator

    def get_twin_state_builder(self) -> TwinStateBuilder:
        if self._twin_state_builder is None:
            self._twin_state_builder = TwinStateBuilder()
        return self._twin_state_builder

    def get_simulation_engine(self) -> EnergySimulationEngine:
        if self._simulation_engine is None:
            self._simulation_engine = load_simulation_engine(
                impact_calculator=self.get_impact_calculator()
            )
        return self._simulation_engine

    def get_recommendation_engine(self) -> RecommendationEngine:
        if self._recommendation_engine is None:
            self._recommendation_engine = load_recommendation_engine()
        return self._recommendation_engine


# Global container instance
_container = ServiceContainer()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Dependency Injection Providers
# ─────────────────────────────────────────────────────────────────────────────

def get_data_loader() -> DataLoader:
    return _container.get_loader()


def get_forecaster() -> EnergyForecaster:
    return _container.get_forecaster()


def get_anomaly_analyzer() -> EnergyAnomalyAnalyzer:
    return _container.get_anomaly_analyzer()


def get_anomaly_detector() -> AnomalyDetector:
    return _container.get_anomaly_detector()


def get_impact_calculator() -> EnergyImpactCalculator:
    return _container.get_impact_calculator()


def get_twin_state_builder() -> TwinStateBuilder:
    return _container.get_twin_state_builder()


def get_simulation_engine() -> EnergySimulationEngine:
    return _container.get_simulation_engine()


def get_recommendation_engine() -> RecommendationEngine:
    return _container.get_recommendation_engine()


def validate_building_and_meter(
    loader: DataLoader,
    building_id: int,
    meter: int = 0,
) -> pd.Timestamp:
    """Validate building_id and meter, returning the latest recorded timestamp.

    Raises:
        HTTPException(422): If building_id or meter are negative.
        HTTPException(404): If building_id or meter are not found in dataset.
    """
    if building_id < 0:
        raise HTTPException(
            status_code=422,
            detail=f"building_id cannot be negative, got {building_id}."
        )
    if meter < 0:
        raise HTTPException(
            status_code=422,
            detail=f"meter cannot be negative, got {meter}."
        )

    try:
        latest_ts = loader.get_latest_timestamp(building_id=building_id, meter=meter)
        return latest_ts
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Building ID {building_id} with meter {meter} was not found in dataset."
        )
