"""Smoke tests for Twinergy 2.0 Data Foundation and Model Compatibility."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import DataLoader, load_data
from src.data.schema import LIGHTGBM_FEATURES, ANOMALY_FEATURES
from src.data.validation import run_all_validations


def test_loader_duckdb_and_fallback():
    loader = DataLoader()
    assert loader.data_path.exists(), "Dataset file must exist"

    # Test load with DuckDB pushdown
    df = loader.load(
        building_id=0,
        meter=0,
        start_time="2016-06-01",
        end_time="2016-06-07",
    )
    assert len(df) > 0, "Filtered query should return rows"
    assert (df["building_id"] == 0).all()
    assert (df["meter"] == 0).all()

    # Test fallback loader
    df_fb = loader._load_fallback(
        building_id=0,
        meter=0,
        start_time="2016-06-01",
        end_time="2016-06-07",
    )
    assert len(df_fb) == len(df), "Fallback loader row count should match DuckDB loader"


def test_model_compatibility():
    root = Path(__file__).resolve().parents[1]
    lgbm_path = root / "twinergy_lgbm.pkl"
    iso_path = root / "twinergy_isoforest.pkl"

    with open(lgbm_path, "rb") as f:
        lgbm = pickle.load(f)
    with open(iso_path, "rb") as f:
        iso = pickle.load(f)

    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=100)
    assert len(sample) == 100

    # Test LightGBM prediction
    X_lgbm = sample[LIGHTGBM_FEATURES]
    preds_log = lgbm.predict(X_lgbm)
    assert len(preds_log) == 100
    preds_kwh = np.expm1(preds_log)
    assert np.all(np.isfinite(preds_kwh))
    assert np.all(preds_kwh >= 0)

    # Test Isolation Forest prediction
    X_iso = sample[ANOMALY_FEATURES]
    anom_scores = iso.predict(X_iso)
    assert len(anom_scores) == 100
    assert set(np.unique(anom_scores)).issubset({-1, 1})


def test_validation_suite():
    res = run_all_validations()
    assert res.passed, f"Validation failed: {[r.name for r in res.results if not r.passed]}"


if __name__ == "__main__":
    print("Running test_loader_duckdb_and_fallback...")
    test_loader_duckdb_and_fallback()
    print("✅ test_loader_duckdb_and_fallback passed.")

    print("Running test_model_compatibility...")
    test_model_compatibility()
    print("✅ test_model_compatibility passed.")

    print("Running test_validation_suite...")
    test_validation_suite()
    print("✅ test_validation_suite passed.")

    print("\n🎉 ALL SMOKE TESTS PASSED SUCCESSFULLY!")
