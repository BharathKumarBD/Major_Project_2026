"""Comprehensive test suite for Twinergy 2.0 TwinState data model and service layer.

Verifies:
1. Basic TwinState creation
2. Required identity validation (building_id, meter, timestamp)
3. Invalid timestamp handling
4. Rejection of NaN and infinite numerical values (ValueError)
5. State status mapping: NORMAL, WARNING, CRITICAL, UNAVAILABLE
6. Preservation of anomaly fields
7. Preservation of forecast fields
8. Preservation of impact fields
9. to_dict() serialization and ISO timestamp formatting
10. Safe handling of optional/missing fields
11. Immutability of source DataFrames and dictionaries
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.twin_state import TwinState, TwinStateBuilder, build_twin_state


def test_basic_twin_state_creation():
    """Verify basic TwinState initialization with required identity fields."""
    ts_now = pd.Timestamp("2016-05-15 12:00:00")
    state = TwinState(building_id=1, timestamp=ts_now, meter=0)

    assert state.building_id == 1
    assert state.timestamp == ts_now
    assert state.meter == 0
    assert state.state_status == "UNAVAILABLE"
    assert state.actual_kwh is None
    assert state.forecast_total_cost is None


def test_identity_validation():
    """Verify validation for missing or invalid identity fields."""
    ts_now = pd.Timestamp("2016-05-15 12:00:00")

    # Negative building_id
    try:
        TwinState(building_id=-5, timestamp=ts_now)
        assert False, "Should raise ValueError for negative building_id"
    except ValueError as e:
        assert "negative" in str(e).lower()

    # Bool building_id
    try:
        TwinState(building_id=True, timestamp=ts_now)
        assert False, "Should raise TypeError for boolean building_id"
    except TypeError:
        pass

    # Negative meter
    try:
        TwinState(building_id=1, timestamp=ts_now, meter=-1)
        assert False, "Should raise ValueError for negative meter"
    except ValueError as e:
        assert "negative" in str(e).lower()


def test_invalid_timestamp_validation():
    """Verify validation for invalid timestamp values."""
    # Invalid string
    try:
        TwinState(building_id=1, timestamp="not-a-timestamp")
        assert False, "Should raise ValueError for invalid timestamp"
    except ValueError:
        pass

    # None timestamp
    try:
        TwinState(building_id=1, timestamp=None)
        assert False, "Should raise ValueError for None timestamp"
    except ValueError:
        pass


def test_nan_and_inf_validation():
    """Verify rejection of NaN and infinite float values."""
    ts_now = pd.Timestamp("2016-05-15 12:00:00")

    # NaN actual_kwh
    try:
        TwinState(building_id=1, timestamp=ts_now, actual_kwh=float("nan"))
        assert False, "Should raise ValueError for NaN actual_kwh"
    except ValueError as e:
        assert "nan" in str(e).lower()

    # Inf forecast_total_cost
    try:
        TwinState(building_id=1, timestamp=ts_now, forecast_total_cost=float("inf"))
        assert False, "Should raise ValueError for infinite forecast_total_cost"
    except ValueError as e:
        assert "infinite" in str(e).lower()


def test_state_status_mappings():
    """Verify state status mapping for NORMAL, WARNING, CRITICAL, and UNAVAILABLE."""
    ts = "2016-05-15 12:00:00"

    # NORMAL
    s_normal = build_twin_state(
        building_id=1,
        timestamp=ts,
        anomaly_analysis={"severity": "NORMAL", "actual_kwh": 100.0, "expected_kwh": 102.0},
    )
    assert s_normal.state_status == "NORMAL"

    # WARNING
    s_warn = build_twin_state(
        building_id=1,
        timestamp=ts,
        anomaly_analysis={"severity": "WARNING", "actual_kwh": 140.0, "expected_kwh": 100.0},
    )
    assert s_warn.state_status == "WARNING"

    # CRITICAL
    s_crit = build_twin_state(
        building_id=1,
        timestamp=ts,
        anomaly_analysis={"severity": "CRITICAL", "actual_kwh": 200.0, "expected_kwh": 100.0},
    )
    assert s_crit.state_status == "CRITICAL"

    # UNAVAILABLE (no primary energy or severity)
    s_unavail = build_twin_state(building_id=1, timestamp=ts)
    assert s_unavail.state_status == "UNAVAILABLE"


def test_builder_preserves_service_outputs():
    """Verify TwinStateBuilder preserves anomaly, forecast, and impact outputs."""
    ts = pd.Timestamp("2016-05-15 12:00:00")

    # Anomaly output dict
    anom_dict = {
        "actual_kwh": 150.0,
        "expected_kwh": 100.0,
        "residual": 50.0,
        "deviation_ratio": 0.50,
        "severity": "WARNING",
        "anomaly_type": "HIGH_DEMAND",
        "classification_reason": "High demand detected.",
    }

    # Forecast output DataFrame
    fc_dates = pd.date_range(start="2016-05-15 13:00:00", periods=24, freq="h")
    fc_df = pd.DataFrame({
        "timestamp": fc_dates,
        "predicted_kwh": np.linspace(100.0, 150.0, 24),
        "weather_source": ["dataset_observed"] * 24,
    })

    # Impact output dict
    imp_dict = {
        "total_kwh": 3000.0,
        "total_cost": 24000.0,
        "total_carbon_kg": 2100.0,
        "peak_cost_hour": pd.Timestamp("2016-05-16 12:00:00"),
    }

    state = build_twin_state(
        building_id=1,
        timestamp=ts,
        anomaly_analysis=anom_dict,
        forecast_df=fc_df,
        impact_dict=imp_dict,
    )

    # Assert anomaly fields
    assert state.actual_kwh == 150.0
    assert state.expected_kwh == 100.0
    assert state.residual_kwh == 50.0
    assert state.deviation_ratio == 0.50
    assert state.severity == "WARNING"
    assert state.anomaly_type == "HIGH_DEMAND"
    assert state.classification_reason == "High demand detected."

    # Assert forecast fields
    assert state.forecast_horizon_hours == 24
    assert state.forecast_total_kwh == 3000.0
    assert state.forecast_peak_kwh == 150.0
    assert state.forecast_peak_hour == pd.Timestamp("2016-05-16 12:00:00")
    assert state.weather_source == "dataset_observed"

    # Assert impact fields
    assert state.forecast_total_cost == 24000.0
    assert state.forecast_total_carbon_kg == 2100.0

    # Assert status
    assert state.state_status == "WARNING"


def test_to_dict_serialization():
    """Verify to_dict() returns a JSON-serializable structure with ISO timestamps."""
    ts = pd.Timestamp("2016-05-15 12:00:00")
    peak_ts = pd.Timestamp("2016-05-16 12:00:00")

    state = TwinState(
        building_id=10,
        timestamp=ts,
        meter=0,
        actual_kwh=120.5,
        expected_kwh=100.0,
        residual_kwh=20.5,
        deviation_ratio=0.205,
        severity="NORMAL",
        anomaly_type="NORMAL",
        forecast_horizon_hours=24,
        forecast_peak_hour=peak_ts,
        state_status="NORMAL",
    )

    d = state.to_dict()

    assert d["building_id"] == 10
    assert d["timestamp"] == "2016-05-15T12:00:00"
    assert d["forecast_peak_hour"] == "2016-05-16T12:00:00"
    assert d["state_status"] == "NORMAL"

    # Verify JSON serializability
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_input_immutability():
    """Verify input DataFrames and dictionaries are not mutated by builder."""
    ts = pd.Timestamp("2016-05-15 12:00:00")

    fc_df = pd.DataFrame({
        "timestamp": pd.date_range(start="2016-05-15 13:00:00", periods=24, freq="h"),
        "predicted_kwh": [100.0] * 24,
        "weather_source": ["provided"] * 24,
    })
    fc_df_orig = fc_df.copy()

    imp_dict = {"total_cost": 500.0, "total_carbon_kg": 70.0}
    imp_dict_orig = imp_dict.copy()

    _ = build_twin_state(
        building_id=1,
        timestamp=ts,
        forecast_df=fc_df,
        impact_dict=imp_dict,
    )

    pd.testing.assert_frame_equal(fc_df, fc_df_orig)
    assert imp_dict == imp_dict_orig


if __name__ == "__main__":
    print("Running test_basic_twin_state_creation...")
    test_basic_twin_state_creation()
    print("✅ test_basic_twin_state_creation passed.")

    print("Running test_identity_validation...")
    test_identity_validation()
    print("✅ test_identity_validation passed.")

    print("Running test_invalid_timestamp_validation...")
    test_invalid_timestamp_validation()
    print("✅ test_invalid_timestamp_validation passed.")

    print("Running test_nan_and_inf_validation...")
    test_nan_and_inf_validation()
    print("✅ test_nan_and_inf_validation passed.")

    print("Running test_state_status_mappings...")
    test_state_status_mappings()
    print("✅ test_state_status_mappings passed.")

    print("Running test_builder_preserves_service_outputs...")
    test_builder_preserves_service_outputs()
    print("✅ test_builder_preserves_service_outputs passed.")

    print("Running test_to_dict_serialization...")
    test_to_dict_serialization()
    print("✅ test_to_dict_serialization passed.")

    print("Running test_input_immutability...")
    test_input_immutability()
    print("✅ test_input_immutability passed.")

    print("\n🎉 ALL TWIN STATE TESTS PASSED SUCCESSFULLY!")
