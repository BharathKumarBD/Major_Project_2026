"""Comprehensive test suite for Twinergy 2.0 EnergySimulationEngine what-if simulation service.

Verifies:
1. +10% energy multiplier scenario
2. 10% energy reduction scenario
3. Fixed kWh addition scenario
4. Input forecast DataFrame immutability
5. delta_kwh calculation correctness
6. delta_percent calculation correctness
7. Zero baseline handling without division-by-zero errors
8. Negative energy consumption rejection
9. Invalid multiplier rejection (<= 0)
10. Invalid reduction rejection (< 0 or >= 1)
11. Baseline vs simulated cost integration via EnergyImpactCalculator
12. Baseline vs simulated carbon integration via EnergyImpactCalculator
13. Baseline vs simulated peak-hour calculations
14. NaN and Infinity parameter/data rejection
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.forecasting import load_forecaster
from src.models.impact import load_impact_calculator
from src.models.simulation import EnergySimulationEngine, load_simulation_engine


def _create_sample_forecast_df() -> pd.DataFrame:
    """Helper to create a standard 24-hour forecast DataFrame."""
    dates = pd.date_range(start="2016-05-15 13:00:00", periods=24, freq="h")
    # Base pattern: 100 kWh to 146 kWh
    kwh_vals = [100.0 + 2.0 * i for i in range(24)]
    return pd.DataFrame({"timestamp": dates, "predicted_kwh": kwh_vals})


def test_multiplier_scenario():
    """Verify +10% energy multiplier scenario (factor = 1.10)."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    res = engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=1.10)

    assert res["scenario_type"] == "ENERGY_MULTIPLIER"
    assert res["scenario_value"] == 1.10

    # Total check
    expected_base_total = df_fc["predicted_kwh"].sum()
    expected_sim_total = expected_base_total * 1.10
    np.testing.assert_allclose(res["baseline_total_kwh"], expected_base_total)
    np.testing.assert_allclose(res["simulated_total_kwh"], expected_sim_total)
    np.testing.assert_allclose(res["delta_total_kwh"], expected_sim_total - expected_base_total)
    np.testing.assert_allclose(res["delta_total_percent"], 10.0)

    # Hourly check
    hourly = res["hourly_df"]
    np.testing.assert_allclose(hourly["simulated_kwh"], df_fc["predicted_kwh"] * 1.10)
    np.testing.assert_allclose(hourly["delta_percent"], 10.0)


def test_reduction_scenario():
    """Verify 10% energy reduction scenario (reduction = 0.10)."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    res = engine.simulate(df_fc, scenario_type="ENERGY_REDUCTION", value=0.10)

    expected_base_total = df_fc["predicted_kwh"].sum()
    expected_sim_total = expected_base_total * 0.90
    np.testing.assert_allclose(res["simulated_total_kwh"], expected_sim_total)
    np.testing.assert_allclose(res["delta_total_percent"], -10.0)

    hourly = res["hourly_df"]
    np.testing.assert_allclose(hourly["simulated_kwh"], df_fc["predicted_kwh"] * 0.90)
    np.testing.assert_allclose(hourly["delta_percent"], -10.0)


def test_fixed_addition_scenario():
    """Verify fixed kWh addition scenario (addition = 5.0 kWh/hr)."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    res = engine.simulate(df_fc, scenario_type="FIXED_KWH_ADDITION", value=5.0)

    hourly = res["hourly_df"]
    np.testing.assert_allclose(hourly["simulated_kwh"], df_fc["predicted_kwh"] + 5.0)
    np.testing.assert_allclose(hourly["delta_kwh"], 5.0)
    np.testing.assert_allclose(res["delta_total_kwh"], 5.0 * 24)


def test_input_immutability():
    """Verify input forecast DataFrame remains strictly unmutated."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()
    df_fc_orig = df_fc.copy()

    _ = engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=1.20)

    pd.testing.assert_frame_equal(df_fc, df_fc_orig)


def test_delta_correctness():
    """Verify delta_kwh and delta_percent calculations."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    res = engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=1.50)
    hourly = res["hourly_df"]

    expected_delta_kwh = hourly["simulated_kwh"] - hourly["baseline_kwh"]
    expected_delta_pct = (expected_delta_kwh / hourly["baseline_kwh"]) * 100.0

    pd.testing.assert_series_equal(hourly["delta_kwh"], expected_delta_kwh, check_names=False)
    pd.testing.assert_series_equal(hourly["delta_percent"], expected_delta_pct, check_names=False)


def test_zero_baseline_handling():
    """Verify safe delta_percent calculation when baseline_kwh is 0."""
    engine = load_simulation_engine()
    dates = pd.date_range(start="2016-05-15 13:00:00", periods=5, freq="h")
    df_fc = pd.DataFrame({"timestamp": dates, "predicted_kwh": [0.0, 0.0, 10.0, 20.0, 0.0]})

    res = engine.simulate(df_fc, scenario_type="FIXED_KWH_ADDITION", value=5.0)
    hourly = res["hourly_df"]

    # Zero baseline rows should have delta_percent = 0.0 instead of inf/nan
    assert not hourly["delta_percent"].isnull().any()
    assert not np.isinf(hourly["delta_percent"]).any()
    assert hourly["delta_percent"].iloc[0] == 0.0
    assert hourly["delta_percent"].iloc[1] == 0.0


def test_negative_energy_rejection():
    """Verify rejection when fixed addition results in negative energy."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    # Attempt to subtract 200 kWh when minimum baseline is 100 kWh
    try:
        engine.simulate(df_fc, scenario_type="FIXED_KWH_ADDITION", value=-200.0)
        assert False, "Should raise ValueError for negative resulting energy"
    except ValueError as e:
        assert "negative" in str(e).lower()


def test_invalid_multiplier_rejection():
    """Verify rejection of zero or negative multiplier factor."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    # Zero factor
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=0.0)
        assert False, "Should raise ValueError for 0 multiplier"
    except ValueError as e:
        assert "must be > 0" in str(e).lower()

    # Negative factor
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=-0.5)
        assert False, "Should raise ValueError for negative multiplier"
    except ValueError as e:
        assert "must be > 0" in str(e).lower()


def test_invalid_reduction_rejection():
    """Verify rejection of out-of-bounds reduction fraction."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    # Reduction < 0
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_REDUCTION", value=-0.1)
        assert False, "Should raise ValueError for negative reduction"
    except ValueError as e:
        assert "range [0, 1)" in str(e).lower()

    # Reduction >= 1.0 (100% or more reduction)
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_REDUCTION", value=1.0)
        assert False, "Should raise ValueError for 1.0 reduction"
    except ValueError as e:
        assert "range [0, 1)" in str(e).lower()


def test_cost_and_carbon_integration():
    """Verify baseline vs simulated cost and carbon metrics via EnergyImpactCalculator."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    res = engine.simulate(
        df_fc,
        scenario_type="ENERGY_MULTIPLIER",
        value=1.20,
        tariff_per_kwh=10.0,
        carbon_intensity_kg_per_kwh=0.50,
    )

    base_kwh = df_fc["predicted_kwh"].sum()
    sim_kwh = base_kwh * 1.20

    # Cost check (@ 10.0 / kWh)
    np.testing.assert_allclose(res["baseline_total_cost"], base_kwh * 10.0)
    np.testing.assert_allclose(res["simulated_total_cost"], sim_kwh * 10.0)
    np.testing.assert_allclose(res["delta_total_cost"], (sim_kwh - base_kwh) * 10.0)

    # Carbon check (@ 0.50 kg / kWh)
    np.testing.assert_allclose(res["baseline_total_carbon_kg"], base_kwh * 0.50)
    np.testing.assert_allclose(res["simulated_total_carbon_kg"], sim_kwh * 0.50)
    np.testing.assert_allclose(res["delta_total_carbon_kg"], (sim_kwh - base_kwh) * 0.50)


def test_peak_hour_calculation():
    """Verify peak energy values and timestamps for baseline and simulated."""
    engine = load_simulation_engine()
    dates = pd.date_range(start="2016-05-15 00:00:00", periods=24, freq="h")
    kwhs = [50.0] * 24
    kwhs[14] = 250.0  # Peak at 14:00

    df_fc = pd.DataFrame({"timestamp": dates, "predicted_kwh": kwhs})

    res = engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=1.10)

    assert res["baseline_peak_kwh"] == 250.0
    assert res["simulated_peak_kwh"] == 275.0
    assert res["baseline_peak_hour"] == pd.Timestamp("2016-05-15 14:00:00")
    assert res["simulated_peak_hour"] == pd.Timestamp("2016-05-15 14:00:00")


def test_nan_and_infinity_rejection():
    """Verify rejection of NaN and infinite inputs."""
    engine = load_simulation_engine()
    df_fc = _create_sample_forecast_df()

    # NaN scenario value
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=float("nan"))
        assert False, "Should raise ValueError for NaN value"
    except ValueError as e:
        assert "nan" in str(e).lower()

    # Infinite scenario value
    try:
        engine.simulate(df_fc, scenario_type="ENERGY_MULTIPLIER", value=float("inf"))
        assert False, "Should raise ValueError for infinite value"
    except ValueError as e:
        assert "infinite" in str(e).lower()

    # NaN in forecast predicted_kwh
    df_nan = df_fc.copy()
    df_nan.loc[2, "predicted_kwh"] = np.nan
    try:
        engine.simulate(df_nan, scenario_type="ENERGY_MULTIPLIER", value=1.10)
        assert False, "Should raise ValueError for NaN in predicted_kwh"
    except ValueError as e:
        assert "nan" in str(e).lower()


if __name__ == "__main__":
    print("Running test_multiplier_scenario...")
    test_multiplier_scenario()
    print("✅ test_multiplier_scenario passed.")

    print("Running test_reduction_scenario...")
    test_reduction_scenario()
    print("✅ test_reduction_scenario passed.")

    print("Running test_fixed_addition_scenario...")
    test_fixed_addition_scenario()
    print("✅ test_fixed_addition_scenario passed.")

    print("Running test_input_immutability...")
    test_input_immutability()
    print("✅ test_input_immutability passed.")

    print("Running test_delta_correctness...")
    test_delta_correctness()
    print("✅ test_delta_correctness passed.")

    print("Running test_zero_baseline_handling...")
    test_zero_baseline_handling()
    print("✅ test_zero_baseline_handling passed.")

    print("Running test_negative_energy_rejection...")
    test_negative_energy_rejection()
    print("✅ test_negative_energy_rejection passed.")

    print("Running test_invalid_multiplier_rejection...")
    test_invalid_multiplier_rejection()
    print("✅ test_invalid_multiplier_rejection passed.")

    print("Running test_invalid_reduction_rejection...")
    test_invalid_reduction_rejection()
    print("✅ test_invalid_reduction_rejection passed.")

    print("Running test_cost_and_carbon_integration...")
    test_cost_and_carbon_integration()
    print("✅ test_cost_and_carbon_integration passed.")

    print("Running test_peak_hour_calculation...")
    test_peak_hour_calculation()
    print("✅ test_peak_hour_calculation passed.")

    print("Running test_nan_and_infinity_rejection...")
    test_nan_and_infinity_rejection()
    print("✅ test_nan_and_infinity_rejection passed.")

    print("\n🎉 ALL SIMULATION ENGINE TESTS PASSED SUCCESSFULLY!")
