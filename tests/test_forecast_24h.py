"""Comprehensive test suite for Twinergy 2.0 24-hour recursive forecasting.

Verifies:
- Exact 24 forecast rows
- Hourly consecutive timestamps strictly after latest historical timestamp
- Autoregressive recursive feature construction (lags, rollings, calendar, weather, proxy)
- Strict leakage safety: zero dependence on future actual meter readings
- Error reporting on insufficient (<168h) or discontinuous history
- Error reporting when future weather is unavailable
- Support for provided future weather
- All 22 canonical LightGBM features present and non-null
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.schema import LIGHTGBM_FEATURES
from src.models.forecasting import EnergyForecaster, load_forecaster


def test_forecast_24h_dimensions_and_features():
    """Verify 24 rows, consecutive hourly timestamps, 22 features, and no NaNs."""
    forecaster = load_forecaster()
    loader = DataLoader()

    # By default, forecast_24h chooses the latest available 24h window in the dataset
    max_dataset_ts = loader.get_latest_timestamp(building_id=1, meter=0)
    expected_cutoff = max_dataset_ts - pd.Timedelta(hours=24)

    df_fc = forecaster.forecast_24h(building_id=1, meter=0)

    # 1. Exactly 24 rows
    assert len(df_fc) == 24, f"Expected 24 forecast rows, got {len(df_fc)}"

    # 2. Hourly consecutive timestamps
    time_diffs = df_fc["timestamp"].diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all(), "Forecast timestamps must be consecutive 1-hour intervals"

    # 3. First timestamp is strictly after the latest historical cutoff timestamp
    first_ts = df_fc["timestamp"].iloc[0]
    assert first_ts > expected_cutoff, f"First timestamp {first_ts} must be after cutoff {expected_cutoff}"
    assert first_ts == expected_cutoff + pd.Timedelta(hours=1), "First forecast step must be cutoff + 1 hour"

    # 4. Steps are 1 to 24
    assert df_fc["step"].tolist() == list(range(1, 25))

    # 5. Required 22 features are present
    for feature in LIGHTGBM_FEATURES:
        assert feature in df_fc.columns, f"Feature '{feature}' missing from forecast output"

    # 6. Structured result columns present
    assert "predicted_kwh" in df_fc.columns
    assert "predicted_log" in df_fc.columns
    assert "weather_source" in df_fc.columns

    # 7. No NaNs in predictions or features
    assert not df_fc["predicted_kwh"].isnull().any(), "predicted_kwh must not contain NaNs"
    assert not df_fc["predicted_log"].isnull().any(), "predicted_log must not contain NaNs"
    assert np.all(np.isfinite(df_fc["predicted_kwh"].values)), "Predictions must be finite"
    assert np.all(df_fc["predicted_kwh"].values >= 0.0), "Predicted kWh must be non-negative"

    for feature in LIGHTGBM_FEATURES:
        assert not df_fc[feature].isnull().any(), f"Feature column '{feature}' contains NaNs"


def test_forecast_24h_custom_cutoff():
    """Verify custom cutoff timestamps produce aligned 24-hour horizons."""
    forecaster = load_forecaster()
    custom_cutoff = pd.Timestamp("2016-10-15 12:00:00")

    df_fc = forecaster.forecast_24h(building_id=1, meter=0, cutoff_time=custom_cutoff)

    assert len(df_fc) == 24
    assert df_fc["timestamp"].iloc[0] == pd.Timestamp("2016-10-15 13:00:00")
    assert df_fc["timestamp"].iloc[-1] == pd.Timestamp("2016-10-16 12:00:00")
    assert df_fc["weather_source"].iloc[0] == "dataset_observed"


def test_forecast_24h_recursive_lag_update():
    """Verify that t+1 prediction recursively becomes lag_1hr at t+2."""
    forecaster = load_forecaster()
    df_fc = forecaster.forecast_24h(building_id=1, meter=0)

    # At step 2, lag_1hr must be equal to predicted_log at step 1
    step1_pred_log = df_fc.loc[df_fc["step"] == 1, "predicted_log"].iloc[0]
    step2_lag1 = df_fc.loc[df_fc["step"] == 2, "lag_1hr"].iloc[0]
    np.testing.assert_allclose(step2_lag1, step1_pred_log, rtol=1e-5,
                               err_msg="Step 2 lag_1hr must equal Step 1 predicted_log")

    # At step 3, lag_1hr must be equal to predicted_log at step 2
    step2_pred_log = df_fc.loc[df_fc["step"] == 2, "predicted_log"].iloc[0]
    step3_lag1 = df_fc.loc[df_fc["step"] == 3, "lag_1hr"].iloc[0]
    np.testing.assert_allclose(step3_lag1, step2_pred_log, rtol=1e-5,
                               err_msg="Step 3 lag_1hr must equal Step 2 predicted_log")

    # At step 7, rolling_mean_6hr must be the mean of predicted_log from steps 1 through 6
    steps_1_to_6_preds = df_fc.loc[df_fc["step"].isin(range(1, 7)), "predicted_log"].values
    step7_roll6 = df_fc.loc[df_fc["step"] == 7, "rolling_mean_6hr"].iloc[0]
    np.testing.assert_allclose(step7_roll6, np.mean(steps_1_to_6_preds), rtol=1e-5,
                               err_msg="Step 7 rolling_mean_6hr must equal mean of steps 1-6 predicted_log")

    # At step 24, lag_1hr must equal predicted_log at step 23
    step23_pred_log = df_fc.loc[df_fc["step"] == 23, "predicted_log"].iloc[0]
    step24_lag1 = df_fc.loc[df_fc["step"] == 24, "lag_1hr"].iloc[0]
    np.testing.assert_allclose(step24_lag1, step23_pred_log, rtol=1e-5,
                               err_msg="Step 24 lag_1hr must equal Step 23 predicted_log")


def test_forecast_24h_leakage_safety():
    """Verify forecast does NOT depend on future actual meter readings.

    We compare predictions under two scenarios:
    1. Normal dataset query where future meter readings exist.
    2. A contaminated historical DataFrame where future meter readings
       are artificially corrupted by 10,000x for timestamps > cutoff_time.
    The forecast must produce identical predictions regardless of future actuals.
    """
    forecaster = load_forecaster()
    loader = DataLoader()
    cutoff_time = pd.Timestamp("2016-10-15 12:00:00")

    # 1. Baseline forecast using dataset
    baseline_fc = forecaster.forecast_24h(building_id=1, meter=0, cutoff_time=cutoff_time)

    # 2. Load historical window + 24 future rows
    start_time = cutoff_time - pd.Timedelta(hours=167)
    future_end = cutoff_time + pd.Timedelta(hours=24)
    df_window = loader.load(
        building_id=1,
        meter=0,
        start_time=start_time,
        end_time=future_end,
        columns=["timestamp", "meter_reading", "meter_reading_log", "square_feet", "air_temperature"],
    )

    # Corrupt future actual readings in the DataFrame
    df_window = df_window.astype({"meter_reading": np.float64, "meter_reading_log": np.float64})
    future_mask = df_window["timestamp"] > cutoff_time
    df_window.loc[future_mask, "meter_reading"] = 9999999.0
    df_window.loc[future_mask, "meter_reading_log"] = float(np.log1p(9999999.0))

    # Run forecast passing this DataFrame with explicit cutoff
    leakage_check_fc = forecaster.forecast_24h(
        building_id=1,
        meter=0,
        cutoff_time=cutoff_time,
        history_df=df_window,
    )

    # Predictions must be identical despite 10,000x corrupted future readings
    np.testing.assert_allclose(
        baseline_fc["predicted_kwh"].values,
        leakage_check_fc["predicted_kwh"].values,
        rtol=1e-5,
        err_msg="Leakage detected! Future actual meter_reading corrupted values altered the 24h forecast."
    )


def test_forecast_24h_with_provided_future_weather():
    """Verify support for user-supplied future weather sequence."""
    forecaster = load_forecaster()
    weather_sequence = [18.5 + (i % 5) for i in range(24)]

    df_fc = forecaster.forecast_24h(
        building_id=1,
        meter=0,
        future_weather=weather_sequence,
    )

    assert len(df_fc) == 24
    assert df_fc["weather_source"].iloc[0] == "provided"
    np.testing.assert_allclose(df_fc["air_temperature"].values, weather_sequence)
    assert not df_fc["predicted_kwh"].isnull().any()


def test_forecast_24h_insufficient_history_error():
    """Verify explicit ValueError when history has fewer than 168 hours."""
    forecaster = load_forecaster()
    cutoff = pd.Timestamp("2016-10-15 12:00:00")

    # Construct only 50 hours of history
    short_dates = pd.date_range(end=cutoff, periods=50, freq="h")
    short_history = pd.DataFrame({
        "timestamp": short_dates,
        "meter_reading_log": np.ones(50) * 4.0,
        "square_feet": np.ones(50) * 2720,
    })

    try:
        forecaster.forecast_24h(building_id=1, meter=0, history_df=short_history)
        assert False, "Expected ValueError for short history (<168h)"
    except ValueError as e:
        assert "168" in str(e)


def test_forecast_24h_discontinuous_history_error():
    """Verify explicit ValueError when historical timestamps contain gaps."""
    forecaster = load_forecaster()
    cutoff = pd.Timestamp("2016-10-15 12:00:00")

    # Construct 168 rows but with a 2-hour missing gap
    dates = pd.date_range(end=cutoff, periods=169, freq="h")
    # Drop one intermediate timestamp to create gap
    discontinuous_dates = dates.delete(80)

    bad_history = pd.DataFrame({
        "timestamp": discontinuous_dates,
        "meter_reading_log": np.ones(168) * 4.0,
        "square_feet": np.ones(168) * 2720,
    })

    try:
        forecaster.forecast_24h(building_id=1, meter=0, history_df=bad_history)
        assert False, "Expected ValueError for discontinuous history"
    except ValueError as e:
        assert "Discontinuous" in str(e) or "discontinuity" in str(e).lower()


def test_forecast_24h_missing_future_weather_error():
    """Verify explicit ValueError when dataset lacks weather and none is provided."""
    forecaster = load_forecaster()
    loader = DataLoader()
    max_dataset_ts = loader.get_latest_timestamp(building_id=1, meter=0)

    # Attempting to forecast strictly past the end of dataset without providing future weather
    # Cutoff at 2016-12-31 23:00:00 means forecasting into 2017-01-01 where no weather exists.
    try:
        forecaster.forecast_24h(
            building_id=1,
            meter=0,
            cutoff_time=max_dataset_ts,
            future_weather=None,
        )
        assert False, "Expected ValueError when future weather is unavailable in dataset"
    except ValueError as e:
        assert "weather" in str(e).lower()


if __name__ == "__main__":
    print("Running test_forecast_24h_dimensions_and_features...")
    test_forecast_24h_dimensions_and_features()
    print("✅ test_forecast_24h_dimensions_and_features passed.")

    print("Running test_forecast_24h_custom_cutoff...")
    test_forecast_24h_custom_cutoff()
    print("✅ test_forecast_24h_custom_cutoff passed.")

    print("Running test_forecast_24h_recursive_lag_update...")
    test_forecast_24h_recursive_lag_update()
    print("✅ test_forecast_24h_recursive_lag_update passed.")

    print("Running test_forecast_24h_leakage_safety...")
    test_forecast_24h_leakage_safety()
    print("✅ test_forecast_24h_leakage_safety passed.")

    print("Running test_forecast_24h_with_provided_future_weather...")
    test_forecast_24h_with_provided_future_weather()
    print("✅ test_forecast_24h_with_provided_future_weather passed.")

    print("Running test_forecast_24h_insufficient_history_error...")
    test_forecast_24h_insufficient_history_error()
    print("✅ test_forecast_24h_insufficient_history_error passed.")

    print("Running test_forecast_24h_discontinuous_history_error...")
    test_forecast_24h_discontinuous_history_error()
    print("✅ test_forecast_24h_discontinuous_history_error passed.")

    print("Running test_forecast_24h_missing_future_weather_error...")
    test_forecast_24h_missing_future_weather_error()
    print("✅ test_forecast_24h_missing_future_weather_error passed.")

    print("\n🎉 ALL 24-HOUR FORECAST TESTS PASSED SUCCESSFULLY!")
