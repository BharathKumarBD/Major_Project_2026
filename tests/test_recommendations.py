"""Comprehensive test suite for Twinergy 2.0 RecommendationEngine service.

Verifies:
1. Critical anomaly recommendation generation
2. High-demand anomaly recommendation generation
3. Rapid-spike anomaly recommendation generation
4. Persistent-high demand recommendation generation
5. Low-demand anomaly recommendation generation
6. Normal-state nominal recommendation generation
7. Cost threshold recommendation generation
8. Carbon threshold recommendation generation
9. Strict priority ordering (CRITICAL > HIGH > MEDIUM > LOW > INFO)
10. Recommendation deduplication
11. Data-backed evidence generation
12. Invalid threshold rejection (ValueError for negative values)
13. Rejection of NaN and infinite threshold values
14. Optional simulation context recommendation (labeled as scenario estimate)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.recommendations import (
    Recommendation,
    RecommendationEngine,
    load_recommendation_engine,
)
from src.models.twin_state import TwinState, build_twin_state


def _create_sample_twin_state(**kwargs) -> TwinState:
    """Helper to construct a valid base TwinState."""
    defaults = {
        "building_id": 1,
        "timestamp": pd.Timestamp("2016-05-15 12:00:00"),
        "meter": 0,
        "actual_kwh": 100.0,
        "expected_kwh": 100.0,
        "residual_kwh": 0.0,
        "deviation_ratio": 0.0,
        "severity": "NORMAL",
        "state_status": "NORMAL",
    }
    defaults.update(kwargs)
    return TwinState(**defaults)


def test_critical_anomaly_recommendation():
    """Verify critical anomaly severity triggers a CRITICAL priority recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="CRITICAL",
        state_status="CRITICAL",
        actual_kwh=220.0,
        expected_kwh=100.0,
        deviation_ratio=1.20,
    )

    recs = engine.generate_recommendations(ts)
    assert len(recs) >= 1
    crit_rec = recs[0]

    assert crit_rec.priority == "CRITICAL"
    assert crit_rec.category == "ANOMALY"
    assert "Investigate Critical Energy Deviation" in crit_rec.title
    assert "Review current building loads" in crit_rec.action
    assert "CRITICAL at 2016-05-15" in crit_rec.evidence


def test_high_demand_recommendation():
    """Verify HIGH_DEMAND anomaly_type triggers a HIGH priority recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="WARNING",
        state_status="WARNING",
        anomaly_type="HIGH_DEMAND",
        actual_kwh=140.0,
        expected_kwh=100.0,
        deviation_ratio=0.40,
    )

    recs = engine.generate_recommendations(ts)
    high_recs = [r for r in recs if r.title == "Investigate Elevated Energy Demand"]
    assert len(high_recs) == 1

    rec = high_recs[0]
    assert rec.priority == "HIGH"
    assert rec.category == "ANOMALY"
    assert "140.0 kWh" in rec.evidence


def test_rapid_spike_recommendation():
    """Verify RAPID_SPIKE anomaly_type triggers a HIGH priority spike recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="WARNING",
        state_status="WARNING",
        anomaly_type="RAPID_SPIKE",
        actual_kwh=160.0,
        expected_kwh=100.0,
    )

    recs = engine.generate_recommendations(ts)
    spike_recs = [r for r in recs if r.title == "Investigate Rapid Consumption Spike"]
    assert len(spike_recs) == 1

    rec = spike_recs[0]
    assert rec.priority == "HIGH"
    assert "sudden load changes" in rec.action.lower()


def test_persistent_high_recommendation():
    """Verify PERSISTENT_HIGH anomaly_type triggers a HIGH priority persistent demand recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="WARNING",
        state_status="WARNING",
        anomaly_type="PERSISTENT_HIGH",
        actual_kwh=130.0,
        expected_kwh=100.0,
    )

    recs = engine.generate_recommendations(ts)
    pers_recs = [r for r in recs if r.title == "Review Sustained High Demand"]
    assert len(pers_recs) == 1

    rec = pers_recs[0]
    assert rec.priority == "HIGH"
    assert rec.category == "OPERATIONS"


def test_low_demand_recommendation():
    """Verify LOW_DEMAND anomaly_type triggers a MEDIUM priority low demand recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="WARNING",
        state_status="WARNING",
        anomaly_type="LOW_DEMAND",
        actual_kwh=40.0,
        expected_kwh=100.0,
        deviation_ratio=-0.60,
    )

    recs = engine.generate_recommendations(ts)
    low_recs = [r for r in recs if r.title == "Check Reduced Demand Activity"]
    assert len(low_recs) == 1

    rec = low_recs[0]
    assert rec.priority == "MEDIUM"
    assert rec.category == "OPERATIONS"


def test_normal_state_recommendation():
    """Verify NORMAL state with no anomalies produces an INFO nominal recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state(
        severity="NORMAL",
        state_status="NORMAL",
        actual_kwh=100.0,
        expected_kwh=102.0,
    )

    recs = engine.generate_recommendations(ts)
    assert len(recs) == 1

    rec = recs[0]
    assert rec.priority == "INFO"
    assert rec.category == "ENERGY"
    assert rec.title == "Energy System Nominal"


def test_cost_threshold_recommendation():
    """Verify forecast cost exceeding cost_threshold produces a COST recommendation."""
    engine = RecommendationEngine(cost_threshold=5000.0)
    ts = _create_sample_twin_state(
        forecast_total_cost=8000.0,
    )

    recs = engine.generate_recommendations(ts)
    cost_recs = [r for r in recs if r.category == "COST"]
    assert len(cost_recs) == 1

    rec = cost_recs[0]
    assert rec.priority == "MEDIUM"
    assert "₹8,000.00" in rec.evidence


def test_carbon_threshold_recommendation():
    """Verify forecast carbon exceeding carbon_threshold produces a CARBON recommendation."""
    engine = RecommendationEngine(carbon_threshold=500.0)
    ts = _create_sample_twin_state(
        forecast_total_carbon_kg=750.0,
    )

    recs = engine.generate_recommendations(ts)
    carbon_recs = [r for r in recs if r.category == "CARBON"]
    assert len(carbon_recs) == 1

    rec = carbon_recs[0]
    assert rec.priority == "MEDIUM"
    assert "750.00 kg CO2e" in rec.evidence


def test_priority_ordering():
    """Verify recommendations are ordered strictly: CRITICAL > HIGH > MEDIUM > LOW > INFO."""
    engine = RecommendationEngine(cost_threshold=1000.0, carbon_threshold=100.0)
    ts = _create_sample_twin_state(
        severity="CRITICAL",
        anomaly_type="HIGH_DEMAND",
        forecast_total_cost=5000.0,
        forecast_total_carbon_kg=500.0,
    )
    sim_res = {"delta_total_kwh": -50.0, "delta_total_cost": -400.0}

    recs = engine.generate_recommendations(ts, simulation_result=sim_res)

    priority_map = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4, "INFO": 5}
    ranks = [priority_map[r.priority] for r in recs]

    assert ranks == sorted(ranks), f"Recommendations must be sorted by priority rank, got {ranks}"


def test_deduplication():
    """Verify identical category/title/action recommendations are deduplicated."""
    engine = RecommendationEngine()
    ts = _create_sample_twin_state(
        severity="WARNING",
        anomaly_type="HIGH_DEMAND",
    )

    recs = engine.generate_recommendations(ts)
    unique_keys = set((r.category, r.title, r.action) for r in recs)
    assert len(recs) == len(unique_keys)


def test_evidence_generation():
    """Verify evidence strings reference actual TwinState data points."""
    engine = RecommendationEngine(cost_threshold=100.0)
    ts = _create_sample_twin_state(
        building_id=42,
        timestamp=pd.Timestamp("2016-10-09 14:00:00"),
        actual_kwh=180.5,
        expected_kwh=100.0,
        deviation_ratio=0.805,
        severity="CRITICAL",
        forecast_total_cost=500.0,
    )

    recs = engine.generate_recommendations(ts)
    evidence_text = " ".join(r.evidence for r in recs)

    assert "Building 42" in evidence_text
    assert "180.5 kWh" in evidence_text
    assert "100.0 kWh" in evidence_text
    assert "₹500.00" in evidence_text


def test_invalid_threshold_rejection():
    """Verify negative threshold values raise explicit ValueError."""
    try:
        RecommendationEngine(cost_threshold=-100.0)
        assert False, "Should raise ValueError for negative cost threshold"
    except ValueError as e:
        assert "negative" in str(e).lower()

    try:
        RecommendationEngine(carbon_threshold=-50.0)
        assert False, "Should raise ValueError for negative carbon threshold"
    except ValueError as e:
        assert "negative" in str(e).lower()


def test_nan_and_infinity_rejection():
    """Verify NaN and infinity threshold values raise explicit ValueError."""
    try:
        RecommendationEngine(cost_threshold=float("nan"))
        assert False, "Should raise ValueError for NaN cost threshold"
    except ValueError as e:
        assert "nan" in str(e).lower()

    try:
        RecommendationEngine(carbon_threshold=float("inf"))
        assert False, "Should raise ValueError for infinite carbon threshold"
    except ValueError as e:
        assert "infinite" in str(e).lower()


def test_simulation_context_recommendation():
    """Verify optional simulation context produces a LOW priority scenario estimate recommendation."""
    engine = load_recommendation_engine()
    ts = _create_sample_twin_state()
    sim_res = {
        "scenario_type": "ENERGY_REDUCTION",
        "scenario_value": 0.10,
        "delta_total_kwh": -120.0,
        "delta_total_cost": -960.0,
    }

    recs = engine.generate_recommendations(ts, simulation_result=sim_res)
    sim_recs = [r for r in recs if r.title == "Simulated Scenario Savings Opportunity"]
    assert len(sim_recs) == 1

    rec = sim_recs[0]
    assert rec.priority == "LOW"
    assert "scenario estimate" in rec.evidence.lower()
    assert "120.0 kWh" in rec.evidence
    assert "₹960.00" in rec.evidence


if __name__ == "__main__":
    print("Running test_critical_anomaly_recommendation...")
    test_critical_anomaly_recommendation()
    print("✅ test_critical_anomaly_recommendation passed.")

    print("Running test_high_demand_recommendation...")
    test_high_demand_recommendation()
    print("✅ test_high_demand_recommendation passed.")

    print("Running test_rapid_spike_recommendation...")
    test_rapid_spike_recommendation()
    print("✅ test_rapid_spike_recommendation passed.")

    print("Running test_persistent_high_recommendation...")
    test_persistent_high_recommendation()
    print("✅ test_persistent_high_recommendation passed.")

    print("Running test_low_demand_recommendation...")
    test_low_demand_recommendation()
    print("✅ test_low_demand_recommendation passed.")

    print("Running test_normal_state_recommendation...")
    test_normal_state_recommendation()
    print("✅ test_normal_state_recommendation passed.")

    print("Running test_cost_threshold_recommendation...")
    test_cost_threshold_recommendation()
    print("✅ test_cost_threshold_recommendation passed.")

    print("Running test_carbon_threshold_recommendation...")
    test_carbon_threshold_recommendation()
    print("✅ test_carbon_threshold_recommendation passed.")

    print("Running test_priority_ordering...")
    test_priority_ordering()
    print("✅ test_priority_ordering passed.")

    print("Running test_deduplication...")
    test_deduplication()
    print("✅ test_deduplication passed.")

    print("Running test_evidence_generation...")
    test_evidence_generation()
    print("✅ test_evidence_generation passed.")

    print("Running test_invalid_threshold_rejection...")
    test_invalid_threshold_rejection()
    print("✅ test_invalid_threshold_rejection passed.")

    print("Running test_nan_and_infinity_rejection...")
    test_nan_and_infinity_rejection()
    print("✅ test_nan_and_infinity_rejection passed.")

    print("Running test_simulation_context_recommendation...")
    test_simulation_context_recommendation()
    print("✅ test_simulation_context_recommendation passed.")

    print("\n🎉 ALL RECOMMENDATION ENGINE TESTS PASSED SUCCESSFULLY!")
