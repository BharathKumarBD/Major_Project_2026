"""Twin State data model and aggregation service layer for Twinergy 2.0.

Provides a clean, structured software representation of a building's current energy state
and derived intelligence at a specific reference timestamp.

IMPORTANT SEMANTIC NOTE:
This TwinState is a software representation of a building energy state constructed from
historical dataset slices, ML model outputs, and impact calculators. It does NOT represent
a live physical digital twin connected to real-time hardware telemetry or IoT actuators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

VALID_STATE_STATUSES = {"NORMAL", "WARNING", "CRITICAL", "UNAVAILABLE"}


@dataclass
class TwinState:
    """Software representation of a building's energy state at a specific timestamp.

    Attributes:
        building_id: Target ASHRAE building identifier.
        timestamp: Reference timestamp for this state observation.
        meter: Target meter ID (0 for Electricity).

        actual_kwh: Observed physical energy consumption (kWh).
        expected_kwh: Model expected energy consumption baseline (kWh).
        residual_kwh: Energy deviation (actual_kwh - expected_kwh).
        deviation_ratio: Relative deviation ratio (residual / expected).

        severity: Anomaly severity classification ('NORMAL', 'WARNING', 'CRITICAL').
        anomaly_type: Deterministic anomaly category ('NORMAL', 'HIGH_DEMAND', 'LOW_DEMAND', etc.).
        classification_reason: Human-readable explanation of energy condition.

        forecast_horizon_hours: Duration of forecast horizon in hours (e.g. 24).
        forecast_total_kwh: Cumulative forecasted energy consumption over horizon.
        forecast_peak_kwh: Maximum predicted hourly consumption in forecast horizon.
        forecast_peak_hour: Timestamp of peak forecasted consumption.

        forecast_total_cost: Total forecasted electricity cost (₹).
        forecast_total_carbon_kg: Total forecasted carbon emissions (kg CO2e).

        weather_source: Source of weather data ('dataset_observed', 'provided').
        state_status: Deterministic aggregated building state status ('NORMAL', 'WARNING', 'CRITICAL', 'UNAVAILABLE').
        generated_at: ISO timestamp recording when this TwinState object was instantiated.
    """

    # Identity / Time
    building_id: int
    timestamp: pd.Timestamp
    meter: int = 0

    # Current Energy
    actual_kwh: Optional[float] = None
    expected_kwh: Optional[float] = None
    residual_kwh: Optional[float] = None
    deviation_ratio: Optional[float] = None

    # Anomaly
    severity: Optional[str] = None
    anomaly_type: Optional[str] = None
    classification_reason: Optional[str] = None

    # Forecast
    forecast_horizon_hours: Optional[int] = None
    forecast_total_kwh: Optional[float] = None
    forecast_peak_kwh: Optional[float] = None
    forecast_peak_hour: Optional[pd.Timestamp] = None

    # Impact
    forecast_total_cost: Optional[float] = None
    forecast_total_carbon_kg: Optional[float] = None

    # Metadata / Status
    weather_source: Optional[str] = None
    state_status: str = "UNAVAILABLE"
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def __post_init__(self):
        """Perform validation and normalization after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate identity fields, state_status vocabulary, and numerical integrity."""
        # 1. Identity validation: building_id
        if self.building_id is None:
            raise ValueError("building_id must be provided.")
        if isinstance(self.building_id, bool):
            raise TypeError("building_id must be an integer, got bool.")
        try:
            self.building_id = int(self.building_id)
        except (ValueError, TypeError):
            raise ValueError(f"building_id must be a valid integer, got {self.building_id}")

        if self.building_id < 0:
            raise ValueError(f"building_id cannot be negative, got {self.building_id}")

        # 2. Identity validation: meter
        if self.meter is None:
            raise ValueError("meter must be provided.")
        if isinstance(self.meter, bool):
            raise TypeError("meter must be an integer, got bool.")
        try:
            self.meter = int(self.meter)
        except (ValueError, TypeError):
            raise ValueError(f"meter must be a valid integer, got {self.meter}")

        if self.meter < 0:
            raise ValueError(f"meter cannot be negative, got {self.meter}")

        # 3. Identity validation: timestamp
        if self.timestamp is None:
            raise ValueError("timestamp must be provided.")

        try:
            ts_conv = pd.to_datetime(self.timestamp)
            if pd.isna(ts_conv):
                raise ValueError("timestamp cannot be NaT or invalid datetime.")
            self.timestamp = ts_conv
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Invalid timestamp: {self.timestamp}. Error: {e}")

        # 4. State status validation
        if self.state_status not in VALID_STATE_STATUSES:
            raise ValueError(
                f"Invalid state_status '{self.state_status}'. "
                f"Must be one of {sorted(list(VALID_STATE_STATUSES))}"
            )

        # 5. Numerical integrity validation (reject NaN and Inf for non-None float fields)
        float_fields = {
            "actual_kwh": self.actual_kwh,
            "expected_kwh": self.expected_kwh,
            "residual_kwh": self.residual_kwh,
            "deviation_ratio": self.deviation_ratio,
            "forecast_total_kwh": self.forecast_total_kwh,
            "forecast_peak_kwh": self.forecast_peak_kwh,
            "forecast_total_cost": self.forecast_total_cost,
            "forecast_total_carbon_kg": self.forecast_total_carbon_kg,
        }

        for name, val in float_fields.items():
            if val is not None:
                if isinstance(val, (bool, str)) and not isinstance(val, (int, float)):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        raise TypeError(f"{name} must be numeric, got {type(val)}")
                if not isinstance(val, (int, float, np.number)):
                    raise TypeError(f"{name} must be numeric, got {type(val)}")
                f_val = float(val)
                if math.isnan(f_val) or np.isnan(f_val):
                    raise ValueError(f"{name} cannot be NaN.")
                if math.isinf(f_val) or np.isinf(f_val):
                    raise ValueError(f"{name} cannot be infinite.")

        # Validate forecast_peak_hour if present
        if self.forecast_peak_hour is not None:
            try:
                peak_ts_conv = pd.to_datetime(self.forecast_peak_hour)
                if pd.isna(peak_ts_conv):
                    raise ValueError("forecast_peak_hour cannot be NaT.")
                self.forecast_peak_hour = peak_ts_conv
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                raise ValueError(f"Invalid forecast_peak_hour: {self.forecast_peak_hour}. Error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert TwinState to a clean, JSON-serializable dictionary."""
        return {
            "building_id": self.building_id,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, (pd.Timestamp, datetime))
                else str(self.timestamp)
            ),
            "meter": self.meter,
            "actual_kwh": self.actual_kwh,
            "expected_kwh": self.expected_kwh,
            "residual_kwh": self.residual_kwh,
            "deviation_ratio": self.deviation_ratio,
            "severity": self.severity,
            "anomaly_type": self.anomaly_type,
            "classification_reason": self.classification_reason,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "forecast_total_kwh": self.forecast_total_kwh,
            "forecast_peak_kwh": self.forecast_peak_kwh,
            "forecast_peak_hour": (
                self.forecast_peak_hour.isoformat()
                if isinstance(self.forecast_peak_hour, (pd.Timestamp, datetime))
                else (str(self.forecast_peak_hour) if self.forecast_peak_hour is not None else None)
            ),
            "forecast_total_cost": self.forecast_total_cost,
            "forecast_total_carbon_kg": self.forecast_total_carbon_kg,
            "weather_source": self.weather_source,
            "state_status": self.state_status,
            "generated_at": self.generated_at,
        }


class TwinStateBuilder:
    """Builder service for aggregating existing service outputs into a TwinState."""

    @staticmethod
    def derive_state_status(
        severity: Optional[str] = None,
        has_primary_energy: bool = True,
    ) -> str:
        """Deterministically map anomaly severity to building state_status.

        Rules:
        - severity == 'NORMAL' → 'NORMAL'
        - severity == 'WARNING' → 'WARNING'
        - severity == 'CRITICAL' → 'CRITICAL'
        - missing severity or missing primary energy measurement → 'UNAVAILABLE'
        """
        if not has_primary_energy or severity is None:
            return "UNAVAILABLE"
        sev_upper = str(severity).upper()
        if sev_upper in ("NORMAL", "WARNING", "CRITICAL"):
            return sev_upper
        return "UNAVAILABLE"

    @classmethod
    def build(
        cls,
        building_id: int,
        timestamp: Union[str, pd.Timestamp, datetime],
        meter: int = 0,
        current_energy: Optional[Dict[str, Any]] = None,
        anomaly_analysis: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        forecast_df: Optional[pd.DataFrame] = None,
        impact_dict: Optional[Dict[str, Any]] = None,
    ) -> TwinState:
        """Construct a TwinState object from pre-computed service outputs without rerunning ML models.

        Args:
            building_id: Target building ID.
            timestamp: Reference timestamp for the state.
            meter: Target meter ID (default 0).
            current_energy: Optional dict with 'actual_kwh', 'expected_kwh', etc.
            anomaly_analysis: Optional DataFrame or dict containing anomaly evaluation.
            forecast_df: Optional 24-hour forecast DataFrame.
            impact_dict: Optional dictionary returned by EnergyImpactCalculator.calculate_impact.

        Returns:
            Validated TwinState instance.
        """
        ts = pd.to_datetime(timestamp)

        # Immutability preservation: defensive copies
        anom_copy = anomaly_analysis.copy() if isinstance(anomaly_analysis, (pd.DataFrame, dict)) else None
        fc_copy = forecast_df.copy() if isinstance(forecast_df, pd.DataFrame) else None
        imp_copy = impact_dict.copy() if isinstance(impact_dict, dict) else None
        curr_copy = current_energy.copy() if isinstance(current_energy, dict) else None

        # 1. Current Energy & Anomaly Extraction
        actual_kwh = None
        expected_kwh = None
        residual_kwh = None
        deviation_ratio = None
        severity = None
        anomaly_type = None
        classification_reason = None

        if curr_copy:
            actual_kwh = curr_copy.get("actual_kwh")
            expected_kwh = curr_copy.get("expected_kwh")
            residual_kwh = curr_copy.get("residual_kwh", curr_copy.get("residual"))
            deviation_ratio = curr_copy.get("deviation_ratio")

        if anom_copy is not None:
            if isinstance(anom_copy, pd.DataFrame):
                if len(anom_copy) > 0:
                    if "timestamp" in anom_copy.columns:
                        matching_rows = anom_copy[pd.to_datetime(anom_copy["timestamp"]) == ts]
                        target_row = matching_rows.iloc[0] if len(matching_rows) > 0 else anom_copy.iloc[-1]
                    else:
                        target_row = anom_copy.iloc[-1]

                    actual_kwh = actual_kwh if actual_kwh is not None else target_row.get("actual_kwh")
                    expected_kwh = expected_kwh if expected_kwh is not None else target_row.get("expected_kwh")
                    residual_kwh = residual_kwh if residual_kwh is not None else target_row.get("residual")
                    deviation_ratio = deviation_ratio if deviation_ratio is not None else target_row.get("deviation_ratio")
                    severity = target_row.get("severity")
                    anomaly_type = target_row.get("anomaly_type")
                    classification_reason = target_row.get("classification_reason")
            elif isinstance(anom_copy, dict):
                actual_kwh = actual_kwh if actual_kwh is not None else anom_copy.get("actual_kwh")
                expected_kwh = expected_kwh if expected_kwh is not None else anom_copy.get("expected_kwh")
                residual_kwh = residual_kwh if residual_kwh is not None else anom_copy.get("residual_kwh", anom_copy.get("residual"))
                deviation_ratio = deviation_ratio if deviation_ratio is not None else anom_copy.get("deviation_ratio")
                severity = anom_copy.get("severity")
                anomaly_type = anom_copy.get("anomaly_type")
                classification_reason = anom_copy.get("classification_reason")

        if residual_kwh is None and actual_kwh is not None and expected_kwh is not None:
            residual_kwh = float(actual_kwh) - float(expected_kwh)

        # 2. Forecast Extraction
        forecast_horizon_hours = None
        forecast_total_kwh = None
        forecast_peak_kwh = None
        forecast_peak_hour = None
        weather_source = None

        if fc_copy is not None and len(fc_copy) > 0:
            forecast_horizon_hours = len(fc_copy)
            if "predicted_kwh" in fc_copy.columns:
                forecast_total_kwh = float(fc_copy["predicted_kwh"].sum())
                forecast_peak_kwh = float(fc_copy["predicted_kwh"].max())
                peak_idx = fc_copy["predicted_kwh"].idxmax()
                if "timestamp" in fc_copy.columns:
                    forecast_peak_hour = pd.to_datetime(fc_copy.loc[peak_idx, "timestamp"])
            if "weather_source" in fc_copy.columns:
                weather_source = str(fc_copy["weather_source"].iloc[0])

        # 3. Impact Extraction
        forecast_total_cost = None
        forecast_total_carbon_kg = None

        if imp_copy is not None:
            forecast_total_cost = imp_copy.get("total_cost")
            forecast_total_carbon_kg = imp_copy.get("total_carbon_kg")
            if forecast_total_kwh is None:
                forecast_total_kwh = imp_copy.get("total_kwh")
            if forecast_peak_hour is None:
                forecast_peak_hour = imp_copy.get("peak_cost_hour")

        # 4. State Status derivation
        has_primary = (actual_kwh is not None) or (expected_kwh is not None)
        state_status = cls.derive_state_status(severity=severity, has_primary_energy=has_primary)

        return TwinState(
            building_id=building_id,
            timestamp=ts,
            meter=meter,
            actual_kwh=float(actual_kwh) if actual_kwh is not None else None,
            expected_kwh=float(expected_kwh) if expected_kwh is not None else None,
            residual_kwh=float(residual_kwh) if residual_kwh is not None else None,
            deviation_ratio=float(deviation_ratio) if deviation_ratio is not None else None,
            severity=str(severity) if severity is not None else None,
            anomaly_type=str(anomaly_type) if anomaly_type is not None else None,
            classification_reason=str(classification_reason) if classification_reason is not None else None,
            forecast_horizon_hours=forecast_horizon_hours,
            forecast_total_kwh=float(forecast_total_kwh) if forecast_total_kwh is not None else None,
            forecast_peak_kwh=float(forecast_peak_kwh) if forecast_peak_kwh is not None else None,
            forecast_peak_hour=forecast_peak_hour,
            forecast_total_cost=float(forecast_total_cost) if forecast_total_cost is not None else None,
            forecast_total_carbon_kg=float(forecast_total_carbon_kg) if forecast_total_carbon_kg is not None else None,
            weather_source=weather_source,
            state_status=state_status,
        )


def build_twin_state(
    building_id: int,
    timestamp: Union[str, pd.Timestamp, datetime],
    meter: int = 0,
    current_energy: Optional[Dict[str, Any]] = None,
    anomaly_analysis: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    impact_dict: Optional[Dict[str, Any]] = None,
) -> TwinState:
    """Convenience factory function for TwinStateBuilder.build()."""
    return TwinStateBuilder.build(
        building_id=building_id,
        timestamp=timestamp,
        meter=meter,
        current_energy=current_energy,
        anomaly_analysis=anomaly_analysis,
        forecast_df=forecast_df,
        impact_dict=impact_dict,
    )
