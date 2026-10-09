"""Comprehensive test suite for Twinergy 2.0 anomaly foundation.

Verifies:
1. Normal consumption (residual ~ 0, severity = NORMAL, is_anomaly = 0)
2. Positive/high deviation (actual > expected, severity = WARNING/CRITICAL, direction = HIGH_DEMAND)
3. Negative/low deviation (actual < expected, severity = WARNING/CRITICAL, direction = LOW_DEMAND)
4. Deterministic severity classification (NORMAL <=0.25, WARNING >0.25-0.50, CRITICAL >0.50)
5. Zero/near-zero expected consumption handling (epsilon floor prevents div-by-zero)
6. Auxiliary Isolation Forest signal integration (isoforest_anomaly, isoforest_score)
7. Building-20 training-scope warning and causality documentation presence
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.schema import ANOMALY_FEATURES
from src.models.anomaly import (
    BUILDING_20_ISOFOREST_WARNING,
    CAUSALITY_WARNING,
    AnomalyDetector,
    EnergyAnomalyAnalyzer,
    load_anomaly_analyzer,
    load_anomaly_detector,
)


def test_normal_consumption():
    """Verify normal consumption (within 25% of expected) is classified as NORMAL."""
    analyzer = load_anomaly_analyzer()
    res = analyzer.analyze_point(actual_kwh=100.0, expected_kwh=100.0)

    assert res["residual"] == 0.0
    assert res["deviation_ratio"] == 0.0
    assert res["abs_deviation_ratio"] == 0.0
    assert res["severity"] == "NORMAL"
    assert res["is_anomaly"] == 0
    assert res["anomaly_direction"] == "NOMINAL"

    # 10% deviation -> still NORMAL
    res_10 = analyzer.analyze_point(actual_kwh=110.0, expected_kwh=100.0)
    assert res_10["severity"] == "NORMAL"
    assert res_10["is_anomaly"] == 0


def test_positive_high_deviation():
    """Verify high positive deviation (actual > expected) triggers HIGH_DEMAND anomaly."""
    analyzer = load_anomaly_analyzer()

    # +30% deviation -> WARNING
    res_warn = analyzer.analyze_point(actual_kwh=130.0, expected_kwh=100.0)
    assert res_warn["residual"] == 30.0
    np.testing.assert_allclose(res_warn["deviation_ratio"], 0.30)
    assert res_warn["severity"] == "WARNING"
    assert res_warn["is_anomaly"] == 1
    assert res_warn["anomaly_direction"] == "HIGH_DEMAND"

    # +60% deviation -> CRITICAL
    res_crit = analyzer.analyze_point(actual_kwh=160.0, expected_kwh=100.0)
    assert res_crit["residual"] == 60.0
    np.testing.assert_allclose(res_crit["deviation_ratio"], 0.60)
    assert res_crit["severity"] == "CRITICAL"
    assert res_crit["is_anomaly"] == 1
    assert res_crit["anomaly_direction"] == "HIGH_DEMAND"


def test_negative_low_deviation():
    """Verify negative deviation (actual < expected) triggers LOW_DEMAND anomaly."""
    analyzer = load_anomaly_analyzer()

    # -30% drop -> WARNING LOW_DEMAND
    res_warn = analyzer.analyze_point(actual_kwh=70.0, expected_kwh=100.0)
    assert res_warn["residual"] == -30.0
    np.testing.assert_allclose(res_warn["deviation_ratio"], -0.30)
    assert res_warn["severity"] == "WARNING"
    assert res_warn["is_anomaly"] == 1
    assert res_warn["anomaly_direction"] == "LOW_DEMAND"

    # -70% drop -> CRITICAL LOW_DEMAND
    res_crit = analyzer.analyze_point(actual_kwh=30.0, expected_kwh=100.0)
    assert res_crit["residual"] == -70.0
    np.testing.assert_allclose(res_crit["deviation_ratio"], -0.70)
    assert res_crit["severity"] == "CRITICAL"
    assert res_crit["is_anomaly"] == 1
    assert res_crit["anomaly_direction"] == "LOW_DEMAND"


def test_severity_classification_boundaries():
    """Verify exact boundary behavior for NORMAL (<=0.25), WARNING (>0.25-0.50), CRITICAL (>0.50)."""
    analyzer = load_anomaly_analyzer()

    # 25% boundary -> NORMAL
    res_25 = analyzer.analyze_point(actual_kwh=125.0, expected_kwh=100.0)
    assert res_25["severity"] == "NORMAL"

    # 25.1% -> WARNING
    res_251 = analyzer.analyze_point(actual_kwh=125.1, expected_kwh=100.0)
    assert res_251["severity"] == "WARNING"

    # 50% boundary -> WARNING
    res_50 = analyzer.analyze_point(actual_kwh=150.0, expected_kwh=100.0)
    assert res_50["severity"] == "WARNING"

    # 50.1% -> CRITICAL
    res_501 = analyzer.analyze_point(actual_kwh=150.1, expected_kwh=100.0)
    assert res_501["severity"] == "CRITICAL"


def test_zero_near_zero_expected_consumption():
    """Verify zero or near-zero expected consumption does not cause DivisionByZero or NaNs."""
    analyzer = load_anomaly_analyzer()

    # expected = 0.0, actual = 5.0 kWh
    res_zero = analyzer.analyze_point(actual_kwh=5.0, expected_kwh=0.0)
    assert np.isfinite(res_zero["deviation_ratio"])
    assert res_zero["severity"] == "CRITICAL"
    assert res_zero["is_anomaly"] == 1

    # DataFrame with expected = 0.0
    df_zero = pd.DataFrame({
        "actual_kwh": [0.0, 5.0, 0.0],
        "expected_kwh": [0.0, 0.0, 0.0],
    })
    analyzed_df = analyzer.analyze(df_zero)
    assert not analyzed_df["deviation_ratio"].isnull().any()
    assert np.all(np.isfinite(analyzed_df["deviation_ratio"].values))


def test_auxiliary_isoforest_signal():
    """Verify Isolation Forest signal is returned as auxiliary columns without overriding primary decision."""
    detector = load_anomaly_detector()
    loader = DataLoader()
    sample = loader.load(building_id=1, meter=0, limit=10)

    analyzed = detector.detect(sample)

    # Primary residual columns present
    assert "residual" in analyzed.columns
    assert "deviation_ratio" in analyzed.columns
    assert "severity" in analyzed.columns
    assert "is_anomaly" in analyzed.columns

    # Auxiliary Isolation Forest columns present
    assert "isoforest_anomaly" in analyzed.columns
    assert "isoforest_score" in analyzed.columns
    assert "isoforest_warning" in analyzed.columns
    assert analyzed["isoforest_warning"].iloc[0] == BUILDING_20_ISOFOREST_WARNING


def test_building20_warning_and_causality_documentation():
    """Verify explicit Building 20 scope warning and non-causality notes are exposed."""
    detector = load_anomaly_detector()
    meta = detector.get_model_metadata()

    assert "Building 20" in meta["training_scope"]
    assert "Building 20" in meta["warning"]
    assert "causal" in meta["causality_note"].lower()
    assert BUILDING_20_ISOFOREST_WARNING in detector.TRAINING_SCOPE_NOTE


if __name__ == "__main__":
    print("Running test_normal_consumption...")
    test_normal_consumption()
    print("✅ test_normal_consumption passed.")

    print("Running test_positive_high_deviation...")
    test_positive_high_deviation()
    print("✅ test_positive_high_deviation passed.")

    print("Running test_negative_low_deviation...")
    test_negative_low_deviation()
    print("✅ test_negative_low_deviation passed.")

    print("Running test_severity_classification_boundaries...")
    test_severity_classification_boundaries()
    print("✅ test_severity_classification_boundaries passed.")

    print("Running test_zero_near_zero_expected_consumption...")
    test_zero_near_zero_expected_consumption()
    print("✅ test_zero_near_zero_expected_consumption passed.")

    print("Running test_auxiliary_isoforest_signal...")
    test_auxiliary_isoforest_signal()
    print("✅ test_auxiliary_isoforest_signal passed.")

    print("Running test_building20_warning_and_causality_documentation...")
    test_building20_warning_and_causality_documentation()
    print("✅ test_building20_warning_and_causality_documentation passed.")

    print("\n🎉 ALL ANOMALY FOUNDATION TESTS PASSED!")
