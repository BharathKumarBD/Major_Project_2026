import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.schema import ANOMALY_FEATURES, LIGHTGBM_FEATURES
from src.models.anomaly import AnomalyDetector, load_anomaly_detector
from src.models.explainability import ModelExplainer, load_explainer
from src.models.forecasting import EnergyForecaster, load_forecaster


def test_forecaster_loading_and_prediction():
    forecaster = load_forecaster()
    assert forecaster.model is not None, "LightGBM model should be loaded"
    assert len(forecaster.feature_names) == 22, "Should preserve exact 22 features"

    # Load 5 sample rows using DataLoader
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=5)
    assert len(sample) == 5

    # Test prediction in kWh
    preds_kwh = forecaster.predict(sample, return_kwh=True)
    assert len(preds_kwh) == 5
    assert np.all(np.isfinite(preds_kwh))
    assert np.all(preds_kwh >= 0.0)

    # Test prediction on log scale
    preds_log = forecaster.predict(sample, return_kwh=False)
    assert len(preds_log) == 5
    np.testing.assert_allclose(preds_kwh, np.expm1(preds_log), rtol=1e-5)

    # Test tree feature importance extraction
    imp_df = forecaster.get_feature_importance(importance_type="gain")
    assert len(imp_df) == 22
    assert list(imp_df.columns) == ["feature", "importance", "importance_type"]
    assert imp_df["importance"].iloc[0] >= imp_df["importance"].iloc[-1]

    # Test error on missing features
    bad_df = sample.drop(columns=["hour"])
    try:
        forecaster.predict(bad_df)
        assert False, "Should raise ValueError when features are missing"
    except ValueError as e:
        assert "hour" in str(e)


def test_anomaly_detector_loading_and_prediction():
    detector = load_anomaly_detector()
    assert detector.model is not None, "Isolation Forest model should be loaded"
    assert len(detector.feature_names) == 8, "Should preserve exact 8 features"

    # Verify Building 20 domain documentation
    meta = detector.get_model_metadata()
    assert "Building 20" in meta["warning"]

    # Load sample rows
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=5)
    assert len(sample) == 5

    # Test raw and binary predictions
    raw_preds = detector.predict_raw(sample)
    assert set(np.unique(raw_preds)).issubset({-1, 1})

    bin_preds = detector.predict(sample)
    assert set(np.unique(bin_preds)).issubset({0, 1})
    np.testing.assert_array_equal(bin_preds, (raw_preds == -1).astype(int))

    # Test full detect pipeline
    enriched = detector.detect(sample)
    assert "is_anomaly" in enriched.columns
    assert "anomaly_score" in enriched.columns
    assert "severity" in enriched.columns


def test_explainability_service():
    explainer = load_explainer()

    # Test global tree importance (not SHAP)
    tree_imp = explainer.get_tree_feature_importance(importance_type="gain")
    assert len(tree_imp) == 22
    assert "feature" in tree_imp.columns
    assert "importance" in tree_imp.columns

    # Test local SHAP explanation on a single row
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=2)

    instance_exp = explainer.explain_instance(sample.iloc[0])
    assert "base_value" in instance_exp
    assert "predicted_log" in instance_exp
    assert "contributions" in instance_exp
    assert len(instance_exp["contributions"]) == 22

    # Verify additive property: base_value + sum(shap_values) == predicted_log
    shap_sum = sum(c["shap_value"] for c in instance_exp["contributions"])
    reconstructed = instance_exp["base_value"] + shap_sum
    np.testing.assert_allclose(reconstructed, instance_exp["predicted_log"], rtol=1e-4)

    # Test bounded sample explanation
    sample_exp = explainer.explain_sample(sample, max_samples=10)
    assert sample_exp["sample_size"] == 2
    assert "mean_abs_shap" in sample_exp
    assert len(sample_exp["mean_abs_shap"]) == 22


def test_forecaster_24h_recursive():
    forecaster = load_forecaster()
    loader = DataLoader()
    max_ts = loader.get_latest_timestamp(building_id=1, meter=0)
    cutoff = max_ts - pd.Timedelta(hours=24)

    fc_df = forecaster.forecast_24h(building_id=1, meter=0)
    assert len(fc_df) == 24, "Must return exactly 24 rows"
    assert fc_df["timestamp"].iloc[0] == cutoff + pd.Timedelta(hours=1)
    assert fc_df["timestamp"].iloc[-1] == max_ts
    assert (fc_df["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert all(feat in fc_df.columns for feat in LIGHTGBM_FEATURES)
    assert not fc_df["predicted_kwh"].isnull().any()
    assert np.all(fc_df["predicted_kwh"].values >= 0.0)


if __name__ == "__main__":
    print("Running test_forecaster_loading_and_prediction...")
    test_forecaster_loading_and_prediction()
    print("✅ test_forecaster_loading_and_prediction passed.")

    print("Running test_forecaster_24h_recursive...")
    test_forecaster_24h_recursive()
    print("✅ test_forecaster_24h_recursive passed.")

    print("Running test_anomaly_detector_loading_and_prediction...")
    test_anomaly_detector_loading_and_prediction()
    print("✅ test_anomaly_detector_loading_and_prediction passed.")

    print("Running test_explainability_service...")
    test_explainability_service()
    print("✅ test_explainability_service passed.")

    print("\n🎉 ALL MODEL SMOKE TESTS PASSED!")
