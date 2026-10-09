"""Energy Impact Calculator service for Twinergy 2.0.

Provides cost and carbon emission forecasting service layer based on the
validated 24-hour energy forecast (predicted_kwh).

Calculations:
- Hourly Cost = predicted_kwh * tariff_per_kwh
- Hourly Carbon (kg CO2e) = predicted_kwh * carbon_intensity_kg_per_kwh

Default assumptions (configurable):
- Default tariff: ₹8.0 / kWh
- Default grid carbon intensity: 0.70 kg CO2e / kWh
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

DEFAULT_TARIFF_PER_KWH: float = 8.0
DEFAULT_CARBON_INTENSITY_KG_PER_KWH: float = 0.70


class EnergyImpactCalculator:
    """Service layer for converting energy forecast (kWh) to cost and carbon metrics."""

    def __init__(
        self,
        tariff_per_kwh: float = DEFAULT_TARIFF_PER_KWH,
        carbon_intensity_kg_per_kwh: float = DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    ):
        """Initialize EnergyImpactCalculator with configurable parameters.

        Args:
            tariff_per_kwh: Electricity tariff per kWh (e.g. 8.0). Must be >= 0.
            carbon_intensity_kg_per_kwh: Carbon intensity in kg CO2e/kWh (e.g. 0.70). Must be >= 0.
        """
        self.tariff_per_kwh = self._validate_value(tariff_per_kwh, "tariff_per_kwh")
        self.carbon_intensity_kg_per_kwh = self._validate_value(
            carbon_intensity_kg_per_kwh, "carbon_intensity_kg_per_kwh"
        )

    def _validate_value(self, val: Any, name: str) -> float:
        """Validate numeric parameter (non-negative, non-NaN, non-infinite)."""
        if isinstance(val, (bool, str)) and not isinstance(val, (int, float)):
            try:
                val = float(val)
            except (ValueError, TypeError):
                raise TypeError(f"{name} must be a valid numeric value, got {type(val)}")

        if not isinstance(val, (int, float, np.number)):
            raise TypeError(f"{name} must be numeric, got {type(val)}")

        f_val = float(val)

        if math.isnan(f_val) or np.isnan(f_val):
            raise ValueError(f"{name} cannot be NaN.")

        if math.isinf(f_val) or np.isinf(f_val):
            raise ValueError(f"{name} cannot be infinite.")

        if f_val < 0.0:
            raise ValueError(f"{name} cannot be negative. Got {f_val}")

        return f_val

    def calculate_hourly(self, energy_kwh: float) -> Dict[str, float]:
        """Calculate single hourly cost and carbon values from energy kWh."""
        kwh = max(0.0, float(energy_kwh))
        cost = kwh * self.tariff_per_kwh
        carbon_kg = kwh * self.carbon_intensity_kg_per_kwh
        return {
            "predicted_kwh": kwh,
            "hourly_cost": cost,
            "hourly_carbon_kg": carbon_kg,
        }

    def calculate_impact(
        self,
        forecast_df: pd.DataFrame,
        tariff_per_kwh: Optional[float] = None,
        carbon_intensity_kg_per_kwh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert 24-hour energy forecast DataFrame into cost and carbon impacts.

        Args:
            forecast_df: DataFrame containing at least 'timestamp' and 'predicted_kwh'.
            tariff_per_kwh: Optional override for electricity tariff.
            carbon_intensity_kg_per_kwh: Optional override for carbon intensity.

        Returns:
            Structured dict containing:
                - 'hourly_df': Copy of forecast_df enriched with 'hourly_cost' and 'hourly_carbon_kg'
                - 'total_kwh': Sum of 24h energy forecast
                - 'total_cost': Sum of 24h cost forecast
                - 'total_carbon_kg': Sum of 24h carbon forecast (kg CO2e)
                - 'average_hourly_cost': Mean hourly cost
                - 'average_hourly_carbon_kg': Mean hourly carbon (kg CO2e)
                - 'peak_cost_hour': Timestamp of peak hourly cost
                - 'peak_carbon_hour': Timestamp of peak hourly carbon
                - 'tariff_per_kwh': Tariff rate used
                - 'carbon_intensity_kg_per_kwh': Carbon intensity rate used
        """
        if "predicted_kwh" not in forecast_df.columns:
            raise ValueError("forecast_df must contain a 'predicted_kwh' column.")

        tariff = (
            self.tariff_per_kwh
            if tariff_per_kwh is None
            else self._validate_value(tariff_per_kwh, "tariff_per_kwh")
        )
        carbon_rate = (
            self.carbon_intensity_kg_per_kwh
            if carbon_intensity_kg_per_kwh is None
            else self._validate_value(carbon_intensity_kg_per_kwh, "carbon_intensity_kg_per_kwh")
        )

        # Do not mutate input DataFrame
        out_df = forecast_df.copy()

        kwh_vals = out_df["predicted_kwh"].values.astype(np.float64)
        if np.isnan(kwh_vals).any():
            raise ValueError("predicted_kwh contains NaN values.")

        hourly_cost = kwh_vals * tariff
        hourly_carbon = kwh_vals * carbon_rate

        out_df["hourly_cost"] = hourly_cost
        out_df["hourly_carbon_kg"] = hourly_carbon

        total_kwh = float(np.sum(kwh_vals))
        total_cost = float(np.sum(hourly_cost))
        total_carbon_kg = float(np.sum(hourly_carbon))

        avg_hourly_cost = float(np.mean(hourly_cost)) if len(out_df) > 0 else 0.0
        avg_hourly_carbon_kg = float(np.mean(hourly_carbon)) if len(out_df) > 0 else 0.0

        if "timestamp" in out_df.columns and len(out_df) > 0:
            peak_idx = int(np.argmax(hourly_cost))
            peak_cost_hour = out_df["timestamp"].iloc[peak_idx]
            peak_carbon_idx = int(np.argmax(hourly_carbon))
            peak_carbon_hour = out_df["timestamp"].iloc[peak_carbon_idx]
        else:
            peak_cost_hour = None
            peak_carbon_hour = None

        return {
            "hourly_df": out_df,
            "total_kwh": total_kwh,
            "total_cost": total_cost,
            "total_carbon_kg": total_carbon_kg,
            "average_hourly_cost": avg_hourly_cost,
            "average_hourly_carbon_kg": avg_hourly_carbon_kg,
            "peak_cost_hour": peak_cost_hour,
            "peak_carbon_hour": peak_carbon_hour,
            "tariff_per_kwh": tariff,
            "carbon_intensity_kg_per_kwh": carbon_rate,
        }


def load_impact_calculator(
    tariff_per_kwh: float = DEFAULT_TARIFF_PER_KWH,
    carbon_intensity_kg_per_kwh: float = DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
) -> EnergyImpactCalculator:
    """Convenience factory function to instantiate EnergyImpactCalculator."""
    return EnergyImpactCalculator(
        tariff_per_kwh=tariff_per_kwh,
        carbon_intensity_kg_per_kwh=carbon_intensity_kg_per_kwh,
    )
