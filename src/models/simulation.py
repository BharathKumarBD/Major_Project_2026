"""What-If Simulation Engine for Twinergy 2.0.

Provides deterministic simulation of 24-hour energy demand scenarios
(ENERGY_MULTIPLIER, ENERGY_REDUCTION, FIXED_KWH_ADDITION) and calculates
the resulting energy, cost, and carbon impact using EnergyImpactCalculator.

NOTE: This simulation engine operates purely on pre-computed forecast DataFrames.
It does NOT rerun LightGBM or recursive forecasting models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.models.impact import (
    DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    DEFAULT_TARIFF_PER_KWH,
    EnergyImpactCalculator,
    load_impact_calculator,
)

VALID_SCENARIO_TYPES = {
    "ENERGY_MULTIPLIER",
    "ENERGY_REDUCTION",
    "FIXED_KWH_ADDITION",
    "OCCUPANCY_CHANGE",
    "TEMPERATURE_CHANGE",
    "EFFICIENCY_MULTIPLIER",
}


class EnergySimulationEngine:
    """Service layer for running deterministic what-if energy demand simulations."""

    def __init__(self, impact_calculator: Optional[EnergyImpactCalculator] = None):
        """Initialize EnergySimulationEngine.

        Args:
            impact_calculator: Optional EnergyImpactCalculator instance for cost/carbon metrics.
        """
        self.impact_calculator = (
            impact_calculator if impact_calculator is not None else load_impact_calculator()
        )

    def _validate_numeric(self, val: Any, name: str) -> float:
        """Validate numeric inputs to reject NaN, Infinity, or invalid types."""
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

        return f_val

    def simulate(
        self,
        forecast_df: pd.DataFrame,
        scenario_type: str,
        value: float,
        tariff_per_kwh: Optional[float] = None,
        carbon_intensity_kg_per_kwh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run what-if energy demand simulation on a 24-hour forecast DataFrame.

        Args:
            forecast_df: DataFrame containing 'timestamp' and 'predicted_kwh'.
            scenario_type: One of 'ENERGY_MULTIPLIER', 'ENERGY_REDUCTION', 'FIXED_KWH_ADDITION'.
            value: Scenario magnitude (multiplier factor, reduction fraction, or fixed kWh addition).
            tariff_per_kwh: Optional override for electricity tariff.
            carbon_intensity_kg_per_kwh: Optional override for carbon intensity.

        Returns:
            Dictionary containing:
                - 'hourly_df': DataFrame with timestamp, baseline_kwh, simulated_kwh, delta_kwh, delta_percent
                - Energy totals & peaks (baseline vs simulated + deltas)
                - Cost totals (baseline vs simulated + delta)
                - Carbon totals (baseline vs simulated + delta)
                - Scenario metadata

        Raises:
            ValueError: If input DataFrame is invalid, contains NaNs/Infs, or parameter bounds are violated.
        """
        if "predicted_kwh" not in forecast_df.columns or "timestamp" not in forecast_df.columns:
            raise ValueError("forecast_df must contain 'timestamp' and 'predicted_kwh' columns.")

        if scenario_type not in VALID_SCENARIO_TYPES:
            raise ValueError(
                f"Invalid scenario_type '{scenario_type}'. Must be one of {sorted(list(VALID_SCENARIO_TYPES))}"
            )

        val = self._validate_numeric(value, "value")

        # Immutability: copy original forecast DataFrame
        out_df = forecast_df.copy()

        baseline_kwh = out_df["predicted_kwh"].values.astype(np.float64)
        if np.isnan(baseline_kwh).any():
            raise ValueError("forecast_df['predicted_kwh'] contains NaN values.")
        if np.isinf(baseline_kwh).any():
            raise ValueError("forecast_df['predicted_kwh'] contains infinite values.")
        if (baseline_kwh < 0.0).any():
            raise ValueError("forecast_df['predicted_kwh'] contains negative energy values.")

        # Scenario Transformation Logic
        if scenario_type == "ENERGY_MULTIPLIER":
            if val <= 0.0:
                raise ValueError(f"ENERGY_MULTIPLIER value must be > 0, got {val}")
            simulated_kwh = baseline_kwh * val

        elif scenario_type == "ENERGY_REDUCTION":
            if val < 0.0 or val >= 1.0:
                raise ValueError(f"ENERGY_REDUCTION fraction must be in range [0, 1), got {val}")
            simulated_kwh = baseline_kwh * (1.0 - val)

        elif scenario_type == "FIXED_KWH_ADDITION":
            simulated_kwh = baseline_kwh + val
            if (simulated_kwh < 0.0).any():
                raise ValueError(
                    f"FIXED_KWH_ADDITION of {val} resulted in negative energy consumption."
                )

        elif scenario_type == "EFFICIENCY_MULTIPLIER":
            if val <= 0.0 or val > 2.0:
                raise ValueError(f"EFFICIENCY_MULTIPLIER must be in range (0, 2.0], got {val}")
            simulated_kwh = baseline_kwh * val

        elif scenario_type == "OCCUPANCY_CHANGE":
            # val is occupancy change factor (e.g. +0.20 for +20%, -0.20 for -20%)
            if val < -0.90 or val > 2.0:
                raise ValueError(f"OCCUPANCY_CHANGE fraction must be in range [-0.90, 2.0], got {val}")
            
            # If occupancy proxy or is_business_hours exists in forecast_df, use it
            if "is_business_hours" in out_df.columns:
                occ_mask = out_df["is_business_hours"].values.astype(float)
            elif "occupancy_proxy" in out_df.columns:
                occ_mask = (out_df["occupancy_proxy"].values > 0).astype(float)
            else:
                # Default heuristic: weekday daytime hours (08:00 to 18:00)
                ts = pd.to_datetime(out_df["timestamp"])
                occ_mask = ((ts.dt.dayofweek < 5) & (ts.dt.hour >= 8) & (ts.dt.hour < 18)).astype(float).values

            # Modulate baseline energy by occupancy shift during active hours (assuming ~40% occupancy sensitivity)
            occupancy_sensitivity = 0.40
            multiplier = 1.0 + (val * occupancy_sensitivity * occ_mask)
            simulated_kwh = baseline_kwh * multiplier

        elif scenario_type == "TEMPERATURE_CHANGE":
            # val is delta temperature in Celsius (e.g. +3.0°C, -2.0°C)
            if val < -15.0 or val > 15.0:
                raise ValueError(f"TEMPERATURE_CHANGE delta must be in range [-15.0, 15.0] °C, got {val}")
            
            # Temperature sensitivity: ~2.5% energy increase per +1°C (HVAC cooling load heuristic)
            temp_sensitivity = 0.025
            multiplier = np.clip(1.0 + (val * temp_sensitivity), 0.1, 3.0)
            simulated_kwh = baseline_kwh * multiplier

        # Delta calculations
        delta_kwh = simulated_kwh - baseline_kwh

        # Safe delta_percent calculation handling baseline == 0
        delta_percent = np.zeros_like(baseline_kwh)
        non_zero_mask = baseline_kwh > 0.0
        delta_percent[non_zero_mask] = (
            delta_kwh[non_zero_mask] / baseline_kwh[non_zero_mask]
        ) * 100.0

        hourly_df = pd.DataFrame({
            "timestamp": out_df["timestamp"].copy(),
            "baseline_kwh": baseline_kwh,
            "simulated_kwh": simulated_kwh,
            "delta_kwh": delta_kwh,
            "delta_percent": delta_percent,
        })

        # Summary Metrics (Energy Totals & Peaks)
        baseline_total_kwh = float(np.sum(baseline_kwh))
        simulated_total_kwh = float(np.sum(simulated_kwh))
        delta_total_kwh = simulated_total_kwh - baseline_total_kwh
        delta_total_percent = (
            (delta_total_kwh / baseline_total_kwh) * 100.0 if baseline_total_kwh > 0.0 else 0.0
        )

        baseline_peak_idx = int(np.argmax(baseline_kwh)) if len(baseline_kwh) > 0 else 0
        simulated_peak_idx = int(np.argmax(simulated_kwh)) if len(simulated_kwh) > 0 else 0

        baseline_peak_kwh = float(baseline_kwh[baseline_peak_idx]) if len(baseline_kwh) > 0 else 0.0
        simulated_peak_kwh = float(simulated_kwh[simulated_peak_idx]) if len(simulated_kwh) > 0 else 0.0

        baseline_peak_hour = out_df["timestamp"].iloc[baseline_peak_idx] if len(out_df) > 0 else None
        simulated_peak_hour = out_df["timestamp"].iloc[simulated_peak_idx] if len(out_df) > 0 else None

        # Cost & Carbon Metrics using EnergyImpactCalculator
        base_df_for_calc = pd.DataFrame({"timestamp": out_df["timestamp"], "predicted_kwh": baseline_kwh})
        sim_df_for_calc = pd.DataFrame({"timestamp": out_df["timestamp"], "predicted_kwh": simulated_kwh})

        base_impact = self.impact_calculator.calculate_impact(
            base_df_for_calc,
            tariff_per_kwh=tariff_per_kwh,
            carbon_intensity_kg_per_kwh=carbon_intensity_kg_per_kwh,
        )
        sim_impact = self.impact_calculator.calculate_impact(
            sim_df_for_calc,
            tariff_per_kwh=tariff_per_kwh,
            carbon_intensity_kg_per_kwh=carbon_intensity_kg_per_kwh,
        )

        baseline_total_cost = base_impact["total_cost"]
        simulated_total_cost = sim_impact["total_cost"]
        delta_total_cost = simulated_total_cost - baseline_total_cost

        baseline_total_carbon_kg = base_impact["total_carbon_kg"]
        simulated_total_carbon_kg = sim_impact["total_carbon_kg"]
        delta_total_carbon_kg = simulated_total_carbon_kg - baseline_total_carbon_kg

        return {
            "hourly_df": hourly_df,
            "scenario_type": scenario_type,
            "scenario_value": val,
            "baseline_total_kwh": baseline_total_kwh,
            "simulated_total_kwh": simulated_total_kwh,
            "delta_total_kwh": delta_total_kwh,
            "delta_total_percent": delta_total_percent,
            "baseline_peak_kwh": baseline_peak_kwh,
            "simulated_peak_kwh": simulated_peak_kwh,
            "baseline_peak_hour": baseline_peak_hour,
            "simulated_peak_hour": simulated_peak_hour,
            "baseline_total_cost": baseline_total_cost,
            "simulated_total_cost": simulated_total_cost,
            "delta_total_cost": delta_total_cost,
            "baseline_total_carbon_kg": baseline_total_carbon_kg,
            "simulated_total_carbon_kg": simulated_total_carbon_kg,
            "delta_total_carbon_kg": delta_total_carbon_kg,
            "tariff_per_kwh": base_impact["tariff_per_kwh"],
            "carbon_intensity_kg_per_kwh": base_impact["carbon_intensity_kg_per_kwh"],
        }


def load_simulation_engine(
    impact_calculator: Optional[EnergyImpactCalculator] = None,
) -> EnergySimulationEngine:
    """Convenience factory function to instantiate EnergySimulationEngine."""
    return EnergySimulationEngine(impact_calculator=impact_calculator)
