"""Comprehensive test suite for Twinergy 2.0 EnergyImpactCalculator service.

Verifies:
1. Default tariff calculation (₹8.0/kWh)
2. Custom tariff calculation
3. Default carbon calculation (0.70 kg CO2e/kWh)
4. Custom carbon intensity calculation
5. Total cost and total carbon emissions aggregation
6. Hourly cost and hourly carbon predictions
7. Zero tariff and zero carbon intensity support
8. Rejection of negative tariffs (ValueError)
9. Rejection of negative carbon intensity (ValueError)
10. Rejection of NaN values (ValueError)
11. Rejection of infinite values (ValueError)
12. Seamless integration with EnergyForecaster.forecast_24h()
13. Preservation of all 24 timestamps in output
14. Immutability of original input forecast DataFrame
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.forecasting import load_forecaster
from src.models.impact import (
    DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    DEFAULT_TARIFF_PER_KWH,
    EnergyImpactCalculator,
    load_impact_calculator,
)


def test_default_impact_calculator():
    """Verify default tariff (8.0) and carbon intensity (0.70)."""
    calc = load_impact_calculator()
    assert calc.tariff_per_kwh == DEFAULT_TARIFF_PER_KWH
    assert calc.carbon_intensity_kg_per_kwh == DEFAULT_CARBON_INTENSITY_KG_PER_KWH

    res = calc.calculate_hourly(10.0)
    assert res["predicted_kwh"] == 10.0
    assert res["hourly_cost"] == 80.0
    assert res["hourly_carbon_kg"] == 7.0


def test_custom_tariff_and_carbon():
    """Verify custom tariff and carbon intensity values."""
    calc = EnergyImpactCalculator(tariff_per_kwh=12.5, carbon_intensity_kg_per_kwh=0.50)
    assert calc.tariff_per_kwh == 12.5
    assert calc.carbon_intensity_kg_per_kwh == 0.50

    res = calc.calculate_hourly(20.0)
    assert res["hourly_cost"] == 250.0
    assert res["hourly_carbon_kg"] == 10.0


def test_zero_tariff_and_carbon():
    """Verify zero tariff and zero carbon intensity are allowed."""
    calc = EnergyImpactCalculator(tariff_per_kwh=0.0, carbon_intensity_kg_per_kwh=0.0)
    assert calc.tariff_per_kwh == 0.0
    assert calc.carbon_intensity_kg_per_kwh == 0.0

    res = calc.calculate_hourly(100.0)
    assert res["hourly_cost"] == 0.0
    assert res["hourly_carbon_kg"] == 0.0


def test_validation_rejects_invalid_parameters():
    """Verify negative, NaN, and infinite inputs raise explicit ValueError."""
    # Negative tariff
    try:
        EnergyImpactCalculator(tariff_per_kwh=-5.0)
        assert False, "Should raise ValueError for negative tariff"
    except ValueError as e:
        assert "negative" in str(e).lower()

    # Negative carbon intensity
    try:
        EnergyImpactCalculator(carbon_intensity_kg_per_kwh=-0.1)
        assert False, "Should raise ValueError for negative carbon intensity"
    except ValueError as e:
        assert "negative" in str(e).lower()

    # NaN tariff
    try:
        EnergyImpactCalculator(tariff_per_kwh=float("nan"))
        assert False, "Should raise ValueError for NaN tariff"
    except ValueError as e:
        assert "nan" in str(e).lower()

    # Infinite tariff
    try:
        EnergyImpactCalculator(tariff_per_kwh=float("inf"))
        assert False, "Should raise ValueError for infinite tariff"
    except ValueError as e:
        assert "infinite" in str(e).lower()


def test_forecast_integration_and_immutability():
    """Verify integration with EnergyForecaster.forecast_24h() output."""
    forecaster = load_forecaster()
    calc = load_impact_calculator()

    # Generate real 24h forecast
    df_fc = forecaster.forecast_24h(building_id=1, meter=0)
    assert len(df_fc) == 24

    # Keep original copy for immutability assertion
    df_fc_orig = df_fc.copy()

    # Calculate 24h impact
    impact = calc.calculate_impact(df_fc)

    # 1. Output structure assertions
    assert "hourly_df" in impact
    assert "total_kwh" in impact
    assert "total_cost" in impact
    assert "total_carbon_kg" in impact
    assert "average_hourly_cost" in impact
    assert "average_hourly_carbon_kg" in impact
    assert "peak_cost_hour" in impact
    assert "peak_carbon_hour" in impact

    # 2. Hourly DataFrame assertions
    out_df = impact["hourly_df"]
    assert len(out_df) == 24
    assert "hourly_cost" in out_df.columns
    assert "hourly_carbon_kg" in out_df.columns

    # Verify timestamps preserved
    pd.testing.assert_series_equal(out_df["timestamp"], df_fc["timestamp"])

    # 3. Numerical relationship assertions
    expected_total_kwh = df_fc["predicted_kwh"].sum()
    np.testing.assert_allclose(impact["total_kwh"], expected_total_kwh)
    np.testing.assert_allclose(impact["total_cost"], expected_total_kwh * DEFAULT_TARIFF_PER_KWH)
    np.testing.assert_allclose(impact["total_carbon_kg"], expected_total_kwh * DEFAULT_CARBON_INTENSITY_KG_PER_KWH)

    # 4. Immutability assertion: original input DataFrame was NOT modified
    assert "hourly_cost" not in df_fc_orig.columns
    pd.testing.assert_frame_equal(df_fc, df_fc_orig)


def test_custom_override_parameters_in_calculate_impact():
    """Verify runtime override of tariff and carbon intensity in calculate_impact."""
    forecaster = load_forecaster()
    calc = load_impact_calculator(tariff_per_kwh=8.0, carbon_intensity_kg_per_kwh=0.70)
    df_fc = forecaster.forecast_24h(building_id=1, meter=0)

    # Override at call time
    impact = calc.calculate_impact(df_fc, tariff_per_kwh=10.0, carbon_intensity_kg_per_kwh=0.50)

    total_kwh = df_fc["predicted_kwh"].sum()
    np.testing.assert_allclose(impact["total_cost"], total_kwh * 10.0)
    np.testing.assert_allclose(impact["total_carbon_kg"], total_kwh * 0.50)
    assert impact["tariff_per_kwh"] == 10.0
    assert impact["carbon_intensity_kg_per_kwh"] == 0.50


if __name__ == "__main__":
    print("Running test_default_impact_calculator...")
    test_default_impact_calculator()
    print("✅ test_default_impact_calculator passed.")

    print("Running test_custom_tariff_and_carbon...")
    test_custom_tariff_and_carbon()
    print("✅ test_custom_tariff_and_carbon passed.")

    print("Running test_zero_tariff_and_carbon...")
    test_zero_tariff_and_carbon()
    print("✅ test_zero_tariff_and_carbon passed.")

    print("Running test_validation_rejects_invalid_parameters...")
    test_validation_rejects_invalid_parameters()
    print("✅ test_validation_rejects_invalid_parameters passed.")

    print("Running test_forecast_integration_and_immutability...")
    test_forecast_integration_and_immutability()
    print("✅ test_forecast_integration_and_immutability passed.")

    print("Running test_custom_override_parameters_in_calculate_impact...")
    test_custom_override_parameters_in_calculate_impact()
    print("✅ test_custom_override_parameters_in_calculate_impact passed.")

    print("\n🎉 ALL ENERGY IMPACT CALCULATOR TESTS PASSED!")
