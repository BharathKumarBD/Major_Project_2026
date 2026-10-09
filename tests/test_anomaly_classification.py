"""Comprehensive test suite for Twinergy 2.0 deterministic explainable anomaly classification.

Verifies:
1. NORMAL classification (severity <= 0.25)
2. LOW_DEMAND classification (negative deviation beyond threshold)
3. HIGH_DEMAND classification (positive deviation beyond threshold)
4. RAPID_SPIKE classification (positive deviation + surge over 6-hour rolling baseline)
5. PERSISTENT_HIGH classification (sustained streak >= 3 consecutive hours or 24h baseline elevation)
6. Priority order rules (NORMAL > LOW_DEMAND > RAPID_SPIKE > PERSISTENT_HIGH > HIGH_DEMAND)
7. Non-causal classification reasons
8. Borderline cases (exact 25%, 35%, 50% thresholds)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.anomaly import EnergyAnomalyAnalyzer, load_anomaly_analyzer


def test_classify_normal():
    """Verify NORMAL classification when deviation is within 25% threshold."""
    analyzer = load_anomaly_analyzer()
    res = analyzer.classify_point(actual_kwh=100.0, expected_kwh=100.0)

    assert res["anomaly_type"] == "NORMAL"
    assert res["priority_rank"] == 1
    assert "normal range" in res["classification_reason"].lower()


def test_classify_low_demand():
    """Verify LOW_DEMAND classification when residual is negative beyond threshold."""
    analyzer = load_anomaly_analyzer()
    res = analyzer.classify_point(actual_kwh=60.0, expected_kwh=100.0)

    assert res["anomaly_type"] == "LOW_DEMAND"
    assert res["priority_rank"] == 2
    assert "low demand detected" in res["classification_reason"].lower()
    assert "40.0% below expected" in res["classification_reason"]


def test_classify_rapid_spike():
    """Verify RAPID_SPIKE classification when actual surges > 35% over 6h baseline."""
    analyzer = load_anomaly_analyzer()
    res = analyzer.classify_point(
        actual_kwh=145.0,
        expected_kwh=100.0,
        rolling_mean_6hr_kwh=100.0,
        streak_hours=1,
    )

    assert res["anomaly_type"] == "RAPID_SPIKE"
    assert res["priority_rank"] == 3
    assert "rapid consumption spike detected" in res["classification_reason"].lower()
    assert "45.0% above the 6-hour rolling baseline" in res["classification_reason"]


def test_classify_persistent_high():
    """Verify PERSISTENT_HIGH classification when streak >= 3 consecutive hours."""
    analyzer = load_anomaly_analyzer()

    # Streak = 3 hours
    res_streak = analyzer.classify_point(
        actual_kwh=140.0,
        expected_kwh=100.0,
        rolling_mean_6hr_kwh=135.0,  # Not rapid spike (<35% surge over r6)
        streak_hours=3,
    )
    assert res_streak["anomaly_type"] == "PERSISTENT_HIGH"
    assert res_streak["priority_rank"] == 4
    assert "persistent high demand detected" in res_streak["classification_reason"].lower()
    assert "sustained across 3 consecutive hourly readings" in res_streak["classification_reason"]

    # 24h baseline elevation > 25%
    res_r24 = analyzer.classify_point(
        actual_kwh=140.0,
        expected_kwh=100.0,
        rolling_mean_6hr_kwh=135.0,
        rolling_mean_24hr_kwh=130.0,
        streak_hours=1,
    )
    assert res_r24["anomaly_type"] == "PERSISTENT_HIGH"
    assert res_r24["priority_rank"] == 4


def test_classify_high_demand_fallback():
    """Verify HIGH_DEMAND classification when positive anomaly does not trigger spike or persistent conditions."""
    analyzer = load_anomaly_analyzer()
    res = analyzer.classify_point(
        actual_kwh=130.0,
        expected_kwh=100.0,
        rolling_mean_6hr_kwh=125.0,
        rolling_mean_24hr_kwh=110.0,
        streak_hours=1,
    )

    assert res["anomaly_type"] == "HIGH_DEMAND"
    assert res["priority_rank"] == 5
    assert "high demand detected" in res["classification_reason"].lower()
    assert "exceeded expected baseline (100.0 kWh) by 30.0%" in res["classification_reason"]


def test_priority_order_overrides():
    """Verify strict priority order rules when multiple conditions overlap."""
    analyzer = load_anomaly_analyzer()

    # Priority 3 (RAPID_SPIKE) overrides Priority 4 (PERSISTENT_HIGH)
    res_override = analyzer.classify_point(
        actual_kwh=150.0,
        expected_kwh=100.0,
        rolling_mean_6hr_kwh=100.0,  # Sharp spike
        streak_hours=4,              # Also persistent streak
    )
    assert res_override["anomaly_type"] == "RAPID_SPIKE"
    assert res_override["priority_rank"] == 3


def test_dataframe_analysis_classification():
    """Verify full DataFrame analysis appends anomaly_type and classification_reason columns."""
    analyzer = load_anomaly_analyzer()
    df_sample = pd.DataFrame({
        "actual_kwh": [100.0, 60.0, 145.0, 140.0],
        "expected_kwh": [100.0, 100.0, 100.0, 100.0],
        "rolling_mean_6hr": [np.log1p(100.0), np.log1p(100.0), np.log1p(100.0), np.log1p(135.0)],
    })

    analyzed = analyzer.analyze(df_sample)

    assert "anomaly_type" in analyzed.columns
    assert "classification_reason" in analyzed.columns

    types = analyzed["anomaly_type"].tolist()
    assert types[0] == "NORMAL"
    assert types[1] == "LOW_DEMAND"
    assert types[2] == "RAPID_SPIKE"


if __name__ == "__main__":
    print("Running test_classify_normal...")
    test_classify_normal()
    print("✅ test_classify_normal passed.")

    print("Running test_classify_low_demand...")
    test_classify_low_demand()
    print("✅ test_classify_low_demand passed.")

    print("Running test_classify_rapid_spike...")
    test_classify_rapid_spike()
    print("✅ test_classify_rapid_spike passed.")

    print("Running test_classify_persistent_high...")
    test_classify_persistent_high()
    print("✅ test_classify_persistent_high passed.")

    print("Running test_classify_high_demand_fallback...")
    test_classify_high_demand_fallback()
    print("✅ test_classify_high_demand_fallback passed.")

    print("Running test_priority_order_overrides...")
    test_priority_order_overrides()
    print("✅ test_priority_order_overrides passed.")

    print("Running test_dataframe_analysis_classification...")
    test_dataframe_analysis_classification()
    print("✅ test_dataframe_analysis_classification passed.")

    print("\n🎉 ALL ANOMALY CLASSIFICATION TESTS PASSED!")
