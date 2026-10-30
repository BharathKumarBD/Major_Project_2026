"""Comprehensive test suite for Twinergy FastAPI Backend Layer.

Verifies:
1. Health endpoint (GET /health) returns 200 and healthy status.
2. Buildings listing (GET /buildings) returns valid building IDs and meter metadata.
3. Meter-filtered buildings (GET /buildings?meter=0).
4. Building digital twin state (GET /buildings/{building_id}/state).
5. State serialization and non-fabrication of missing fields.
6. 24-hour recursive energy forecast (GET /buildings/{building_id}/forecast).
7. Energy anomalies analysis (GET /buildings/{building_id}/anomalies) distinguishing primary residual vs auxiliary isolation forest.
8. Cost & carbon impact calculation (GET /buildings/{building_id}/impact).
9. Energy recommendations (GET /buildings/{building_id}/recommendations) preserving priority ordering.
10. Simulation scenario: ENERGY_MULTIPLIER.
11. Simulation scenario: ENERGY_REDUCTION.
12. Simulation scenario: FIXED_KWH_ADDITION.
13. Simulation immutability: verifies baseline forecast and state are not modified.
14. Error handling: 404 for non-existent building.
15. Error handling: 422 for negative building_id and negative meter.
16. Error handling: 422 for invalid simulation scenario types and out-of-bounds parameters.
17. Error handling: 400 for simulation addition leading to negative energy.
18. Error handling: 400 when forecast/weather data is unavailable.
19. Sanitization: ensures system filesystem paths are not exposed in error details.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.main import app
from src.api.services import sanitize_error_message, to_iso_string
from src.models.twin_state import TwinState


@pytest.fixture(scope="module")
def client():
    """FastAPI test client fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """Verify /health returns 200 OK and expected service metadata without loading dataset."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Twinergy API"
    assert "version" in data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Buildings Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_buildings_all(client):
    """Verify GET /buildings returns list of building IDs and all 4 meter types."""
    response = client.get("/buildings")
    assert response.status_code == 200
    data = response.json()

    assert "total_buildings" in data
    assert "building_ids" in data
    assert "meters" in data
    assert data["total_buildings"] > 0
    assert len(data["building_ids"]) == data["total_buildings"]

    # Verify meters list
    meter_ids = [m["meter_id"] for m in data["meters"]]
    assert 0 in meter_ids  # Electricity
    assert 1 in meter_ids  # Chilled Water


def test_get_buildings_filtered_by_meter(client):
    """Verify GET /buildings?meter=0 returns building IDs for electricity meter."""
    response = client.get("/buildings?meter=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total_buildings"] > 0
    assert 0 in data["building_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Twin State Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_building_state_success(client):
    """Verify GET /buildings/0/state returns complete validated TwinState representation."""
    response = client.get("/buildings/0/state?meter=0")
    assert response.status_code == 200
    data = response.json()

    # Identity fields
    assert data["building_id"] == 0
    assert data["meter"] == 0
    assert "timestamp" in data
    assert "generated_at" in data

    # State status
    assert data["state_status"] in {"NORMAL", "WARNING", "CRITICAL", "UNAVAILABLE"}

    # Current energy fields
    assert data["actual_kwh"] is not None
    assert data["expected_kwh"] is not None
    assert data["residual_kwh"] is not None
    assert data["deviation_ratio"] is not None

    # Forecast and impact fields
    assert data["forecast_horizon_hours"] == 24
    assert data["forecast_total_kwh"] is not None
    assert data["forecast_total_kwh"] > 0
    assert data["forecast_total_cost"] is not None
    assert data["forecast_total_carbon_kg"] is not None


def test_get_building_state_custom_ref_time(client):
    """Verify GET /buildings/0/state with explicit ref_time ISO query parameter."""
    ref_time = "2016-12-30T12:00:00"
    response = client.get(f"/buildings/0/state?meter=0&ref_time={ref_time}")
    assert response.status_code == 200
    data = response.json()
    assert data["timestamp"] == ref_time


def test_get_building_state_handles_unavailable_forecast_without_zero_fabrication(client):
    """Verify TwinState serialization preserves None (null) and does not fabricate zeros when forecast fails."""
    with patch("src.models.forecasting.EnergyForecaster.forecast_24h", side_effect=ValueError("Weather unavailable")):
        response = client.get("/buildings/0/state?meter=0")
        assert response.status_code == 200
        data = response.json()

        # Forecast fields must be None / null, NEVER fabricated zeros
        assert data["forecast_horizon_hours"] is None
        assert data["forecast_total_kwh"] is None
        assert data["forecast_peak_kwh"] is None
        assert data["forecast_total_cost"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forecast Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_building_forecast_success(client):
    """Verify GET /buildings/0/forecast returns 24 hourly rows and proper summary."""
    response = client.get("/buildings/0/forecast?meter=0")
    assert response.status_code == 200
    data = response.json()

    assert data["building_id"] == 0
    assert data["meter"] == 0
    assert "cutoff_time" in data
    assert data["weather_source"] in {"dataset_observed", "provided"}

    # Summary checks
    summary = data["summary"]
    assert summary["total_kwh"] > 0
    assert summary["peak_kwh"] >= summary["min_kwh"]
    assert "peak_timestamp" in summary
    assert "min_timestamp" in summary

    # Hourly predictions checks
    hourly = data["hourly_forecast"]
    assert len(hourly) == 24
    steps = [h["step"] for h in hourly]
    assert steps == list(range(1, 25))
    for h in hourly:
        assert h["predicted_kwh"] >= 0.0
        assert "timestamp" in h
        assert "predicted_log" in h


def test_get_building_forecast_unavailable_weather_returns_400(client):
    """Verify forecast endpoint returns 400 with clean message when future weather is unavailable."""
    with patch("src.models.forecasting.EnergyForecaster.forecast_24h", side_effect=ValueError("Future weather is missing")):
        response = client.get("/buildings/0/forecast?meter=0")
        assert response.status_code == 400
        data = response.json()
        assert "Forecast unavailable" in data["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Anomalies Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_building_anomalies_success(client):
    """Verify GET /buildings/0/anomalies separates primary residual signal and auxiliary Isolation Forest."""
    response = client.get("/buildings/0/anomalies?meter=0&limit=48")
    assert response.status_code == 200
    data = response.json()

    assert data["building_id"] == 0
    assert data["meter"] == 0
    assert data["total_records"] > 0
    assert "anomaly_count" in data

    # 1. Primary residual signal
    primary = data["primary_residual_signal"]
    assert primary["warning_threshold"] == 0.25
    assert primary["critical_threshold"] == 0.50
    assert "counts_by_severity" in primary
    assert "counts_by_type" in primary
    assert "statistical divergence" in primary["causality_warning"].lower()

    # 2. Auxiliary Isolation Forest signal
    aux = data["auxiliary_isolation_forest_signal"]
    assert "Building 20 only" in aux["training_scope"]
    assert "warning" in aux

    # 3. Records list
    anomalies = data["anomalies"]
    assert len(anomalies) > 0
    sample = anomalies[0]
    assert "residual" in sample
    assert "deviation_ratio" in sample
    assert "anomaly_type" in sample
    assert "classification_reason" in sample


def test_get_building_anomalies_filter_only(client):
    """Verify anomalies_only=True returns only records where is_anomaly == 1."""
    response = client.get("/buildings/0/anomalies?meter=0&limit=168&anomalies_only=true")
    assert response.status_code == 200
    data = response.json()
    for item in data["anomalies"]:
        assert item["is_anomaly"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Impact Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_building_impact_success(client):
    """Verify GET /buildings/0/impact computes cost and carbon impacts with default assumptions."""
    response = client.get("/buildings/0/impact?meter=0")
    assert response.status_code == 200
    data = response.json()

    # Assumptions
    assumptions = data["assumptions"]
    assert assumptions["tariff_per_kwh"] == 8.0
    assert assumptions["carbon_intensity_kg_per_kwh"] == 0.70

    # Summary
    summary = data["summary"]
    assert summary["total_kwh"] > 0
    assert summary["total_cost"] > 0
    assert summary["total_carbon_kg"] > 0
    assert summary["average_hourly_cost"] > 0

    # Hourly impact items
    hourly = data["hourly_impact"]
    assert len(hourly) == 24
    for h in hourly:
        assert h["hourly_cost"] >= 0.0
        assert h["hourly_carbon_kg"] >= 0.0


def test_get_building_impact_custom_tariffs(client):
    """Verify custom tariff and carbon intensity parameters are correctly applied."""
    tariff = 12.50
    carbon_rate = 0.85
    response = client.get(f"/buildings/0/impact?meter=0&tariff_per_kwh={tariff}&carbon_intensity_kg_per_kwh={carbon_rate}")
    assert response.status_code == 200
    data = response.json()

    assert data["assumptions"]["tariff_per_kwh"] == tariff
    assert data["assumptions"]["carbon_intensity_kg_per_kwh"] == carbon_rate

    # Cost must equal energy * tariff
    kwh = data["summary"]["total_kwh"]
    expected_cost = kwh * tariff
    assert abs(data["summary"]["total_cost"] - expected_cost) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Recommendations Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_building_recommendations_success(client):
    """Verify GET /buildings/0/recommendations returns ordered advisory recommendations."""
    response = client.get("/buildings/0/recommendations?meter=0")
    assert response.status_code == 200
    data = response.json()

    assert data["building_id"] == 0
    assert data["meter"] == 0
    assert "twin_state_status" in data
    assert "total_count" in data
    assert "recommendations" in data

    recs = data["recommendations"]
    assert data["total_count"] == len(recs)

    # Check priority ordering: CRITICAL -> HIGH -> MEDIUM -> LOW -> INFO
    priority_order = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4, "INFO": 5}
    ranks = [priority_order.get(r["priority"], 99) for r in recs]
    assert ranks == sorted(ranks), "Recommendations must be sorted by priority rank"

    # Check structure of items
    for r in recs:
        assert "recommendation_id" in r
        assert "priority" in r
        assert "category" in r
        assert "title" in r
        assert "action" in r
        assert "evidence" in r


# ─────────────────────────────────────────────────────────────────────────────
# 8. What-If Simulation Endpoint Tests (All 3 Scenario Types)
# ─────────────────────────────────────────────────────────────────────────────

def test_simulate_energy_multiplier(client):
    """Verify POST /buildings/0/simulate with ENERGY_MULTIPLIER (+10% demand)."""
    payload = {
        "scenario_type": "ENERGY_MULTIPLIER",
        "value": 1.10,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["scenario_type"] == "ENERGY_MULTIPLIER"
    assert data["scenario_value"] == 1.10

    summary = data["summary"]
    # Simulated total energy must be ~10% greater than baseline
    np.testing.assert_allclose(summary["delta_total_percent"], 10.0, atol=1e-3)
    assert summary["simulated_total_cost"] > summary["baseline_total_cost"]
    assert summary["simulated_total_carbon_kg"] > summary["baseline_total_carbon_kg"]

    # Hourly items
    hourly = data["hourly_simulation"]
    assert len(hourly) == 24
    for h in hourly:
        np.testing.assert_allclose(h["delta_percent"], 10.0, atol=1e-3)


def test_simulate_energy_reduction(client):
    """Verify POST /buildings/0/simulate with ENERGY_REDUCTION (15% reduction)."""
    payload = {
        "scenario_type": "ENERGY_REDUCTION",
        "value": 0.15,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    summary = data["summary"]
    # Simulated total energy must be -15% less than baseline
    np.testing.assert_allclose(summary["delta_total_percent"], -15.0, atol=1e-3)
    assert summary["delta_total_cost"] < 0
    assert summary["delta_total_carbon_kg"] < 0


def test_simulate_fixed_kwh_addition(client):
    """Verify POST /buildings/0/simulate with FIXED_KWH_ADDITION (+10 kWh per hour)."""
    payload = {
        "scenario_type": "FIXED_KWH_ADDITION",
        "value": 10.0,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    summary = data["summary"]
    # 24 hours * 10 kWh = +240 kWh total delta
    np.testing.assert_allclose(summary["delta_total_kwh"], 240.0, atol=1e-3)


def test_simulation_does_not_modify_baseline_state(client):
    """Verify simulation is read-only and does not mutate baseline TwinState or forecast."""
    # 1. Fetch initial state
    res1 = client.get("/buildings/0/state?meter=0")
    assert res1.status_code == 200
    initial_state = res1.json()

    # 2. Run an aggressive simulation scenario (+50% multiplier)
    sim_res = client.post(
        "/buildings/0/simulate",
        json={"scenario_type": "ENERGY_MULTIPLIER", "value": 1.50, "meter": 0},
    )
    assert sim_res.status_code == 200

    # 3. Fetch state again and verify strict equality
    res2 = client.get("/buildings/0/state?meter=0")
    assert res2.status_code == 200
    post_state = res2.json()

    assert initial_state["actual_kwh"] == post_state["actual_kwh"]
    assert initial_state["expected_kwh"] == post_state["expected_kwh"]
    assert initial_state["forecast_total_kwh"] == post_state["forecast_total_kwh"]
    assert initial_state["forecast_total_cost"] == post_state["forecast_total_cost"]
    assert initial_state["state_status"] == post_state["state_status"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Validation and Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_building_returns_404(client):
    """Verify requests for non-existent building ID return HTTP 404."""
    response = client.get("/buildings/999999/state?meter=0")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_unsupported_meter_returns_404(client):
    """Verify requests for an invalid/unsupported building-meter pair return HTTP 404."""
    response = client.get("/buildings/0/state?meter=3")  # Meter 3 does not exist for building 0
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_negative_building_id_returns_422(client):
    """Verify negative building ID returns HTTP 422 validation error."""
    response = client.get("/buildings/-5/state?meter=0")
    assert response.status_code == 422
    assert "negative" in str(response.json()["detail"]).lower()


def test_negative_meter_returns_422(client):
    """Verify negative meter returns HTTP 422 validation error."""
    response = client.get("/buildings/0/state?meter=-1")
    assert response.status_code == 422


def test_invalid_simulation_scenario_type_returns_422(client):
    """Verify invalid simulation scenario_type returns HTTP 422."""
    payload = {
        "scenario_type": "INVALID_SCENARIO",
        "value": 1.10,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 422


def test_invalid_multiplier_value_returns_422(client):
    """Verify non-positive ENERGY_MULTIPLIER parameter returns HTTP 422."""
    payload = {
        "scenario_type": "ENERGY_MULTIPLIER",
        "value": -0.20,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 422


def test_invalid_reduction_fraction_returns_422(client):
    """Verify reduction fraction >= 1.0 or < 0 returns HTTP 422."""
    payload = {
        "scenario_type": "ENERGY_REDUCTION",
        "value": 1.25,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 422


def test_fixed_kwh_addition_causing_negative_energy_returns_400(client):
    """Verify FIXED_KWH_ADDITION resulting in negative consumption returns HTTP 400."""
    payload = {
        "scenario_type": "FIXED_KWH_ADDITION",
        "value": -999999.0,
        "meter": 0,
    }
    response = client.post("/buildings/0/simulate", json=payload)
    assert response.status_code == 400
    assert "negative energy consumption" in response.json()["detail"]


def test_error_message_path_sanitization():
    """Verify sanitize_error_message removes system paths from error strings."""
    raw_error = "File not found at /Users/sunilkumar/Desktop/Major_Project_2026/twinergy_clean.parquet"
    sanitized = sanitize_error_message(raw_error)
    assert "/Users/sunilkumar" not in sanitized
    assert "<dataset_path>" in sanitized or "not found" in sanitized
