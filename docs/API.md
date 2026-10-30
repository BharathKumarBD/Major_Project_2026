# Twinergy — REST API Documentation (Step 11A)

The Twinergy API is a clean, read-only FastAPI backend layer that exposes existing forecasting, anomaly detection, building digital twin state representations, cost/carbon impact calculations, rule-based recommendations, and what-if energy demand simulations through RESTful HTTP endpoints.

---

## 1. Quick Start

### Start the API Server
Using the project's virtual environment:

```bash
major/bin/python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Interactive Documentation
Once started, explore and test the endpoints directly via your browser:
- **Swagger UI (Interactive):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Schema (JSON):** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### Run Tests
```bash
# Run new API endpoint tests
major/bin/pytest tests/test_api.py -v

# Run the full project test suite (100 tests)
major/bin/pytest tests/
```

---

## 2. Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and version check |
| `GET` | `/buildings` | Available building IDs and supported meter types |
| `GET` | `/buildings/{building_id}/state` | Aggregated software `TwinState` representation |
| `GET` | `/buildings/{building_id}/forecast` | Recursive 24-hour multi-step energy forecast |
| `GET` | `/buildings/{building_id}/anomalies` | Primary residual anomalies and auxiliary Isolation Forest signals |
| `GET` | `/buildings/{building_id}/impact` | 24-hour electricity cost (₹) and carbon emissions (kg CO2e) |
| `GET` | `/buildings/{building_id}/recommendations`| Ordered advisory energy management recommendations |
| `POST` | `/buildings/{building_id}/simulate` | Deterministic what-if scenario demand simulation |

---

## 3. Detailed Endpoint Specifications & Examples

### 3.1 Service Health Check
- **Endpoint:** `GET /health`
- **Purpose:** Fast liveness and readiness probe that does not load the dataset or ML models.
- **Example Request:**
  ```bash
  curl -s http://127.0.0.1:8000/health
  ```
- **Example Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "Twinergy API",
    "version": "2.0.0"
  }
  ```

---

### 3.2 List Available Buildings and Meters
- **Endpoint:** `GET /buildings`
- **Query Parameters:**
  - `meter` *(optional, int)*: Filter building IDs by meter ID (e.g. `0` for Electricity).
- **Purpose:** Returns available building IDs and meter types using DuckDB pushdown queries without loading the full 6.34 GB CSV into memory.
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings?meter=0"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "total_buildings": 1413,
    "building_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "meters": [
      {"meter_id": 0, "name": "Electricity"},
      {"meter_id": 1, "name": "Chilled Water"},
      {"meter_id": 2, "name": "Steam"},
      {"meter_id": 3, "name": "Hot Water"}
    ]
  }
  ```

---

### 3.3 Building Digital Twin State
- **Endpoint:** `GET /buildings/{building_id}/state`
- **Query Parameters:**
  - `meter` *(optional, int, default: 0)*: Target meter identifier.
  - `ref_time` *(optional, str)*: ISO reference timestamp for the state. Defaults to the latest historical timestamp in the dataset.
- **Purpose:** Constructs and returns the validated `TwinState` representation aggregating current consumption, residual anomaly classification, 24-hour forecast totals, and cost/carbon impacts.
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings/0/state?meter=0"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "timestamp": "2016-12-31T23:00:00",
    "meter": 0,
    "actual_kwh": 175.42,
    "expected_kwh": 180.73,
    "residual_kwh": -5.31,
    "deviation_ratio": -0.029,
    "severity": "NORMAL",
    "anomaly_type": "NORMAL",
    "classification_reason": "Consumption is within normal range (relative deviation: -2.9%).",
    "forecast_horizon_hours": 24,
    "forecast_total_kwh": 4646.49,
    "forecast_peak_kwh": 230.06,
    "forecast_peak_hour": "2016-12-31T00:00:00",
    "forecast_total_cost": 37171.88,
    "forecast_total_carbon_kg": 3252.54,
    "weather_source": "dataset_observed",
    "state_status": "NORMAL",
    "generated_at": "2026-10-25T14:52:04Z"
  }
  ```

---

### 3.4 24-Hour Recursive Energy Forecast
- **Endpoint:** `GET /buildings/{building_id}/forecast`
- **Query Parameters:**
  - `meter` *(optional, int, default: 0)*: Target meter identifier.
  - `cutoff_time` *(optional, str)*: Historical cutoff timestamp. The 24-hour forecast begins strictly at `cutoff_time + 1 hour`. Defaults to 24 hours prior to the latest available timestamp.
- **Purpose:** Executes the true recursive 24-hour multi-step forecast using LightGBM (`twinergy_lgbm.pkl`).
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings/0/forecast?meter=0"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "meter": 0,
    "cutoff_time": "2016-12-30T23:00:00",
    "weather_source": "dataset_observed",
    "summary": {
      "total_kwh": 4646.49,
      "peak_kwh": 230.06,
      "peak_timestamp": "2016-12-31T00:00:00",
      "min_kwh": 169.05,
      "min_timestamp": "2016-12-31T14:00:00",
      "average_kwh": 193.60
    },
    "hourly_forecast": [
      {
        "step": 1,
        "timestamp": "2016-12-31T00:00:00",
        "predicted_kwh": 230.06,
        "predicted_log": 5.44,
        "air_temperature": 18.5
      }
    ]
  }
  ```

---

### 3.5 Energy Anomalies Analysis
- **Endpoint:** `GET /buildings/{building_id}/anomalies`
- **Query Parameters:**
  - `meter` *(optional, int, default: 0)*
  - `start_time` *(optional, str)*: Start of window (ISO format)
  - `end_time` *(optional, str)*: End of window (ISO format)
  - `limit` *(optional, int, default: 168)*: Maximum hourly records (1 to 1000)
  - `anomalies_only` *(optional, bool, default: false)*: If true, filters list to `is_anomaly == 1`
- **Purpose:** Evaluates consumption against expected baseline using primary residual detection (5 deterministic classes: `NORMAL`, `HIGH_DEMAND`, `LOW_DEMAND`, `RAPID_SPIKE`, `PERSISTENT_HIGH`), and separates the auxiliary Isolation Forest signal.
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings/0/anomalies?meter=0&limit=24"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "meter": 0,
    "window_start": "2016-12-30T23:00:00",
    "window_end": "2016-12-31T23:00:00",
    "total_records": 25,
    "anomaly_count": 0,
    "primary_residual_signal": {
      "warning_threshold": 0.25,
      "critical_threshold": 0.50,
      "counts_by_severity": {"NORMAL": 25, "WARNING": 0, "CRITICAL": 0},
      "counts_by_type": {"NORMAL": 25},
      "causality_warning": "Residual deviation indicates statistical divergence from expected baseline, not physical causal proof."
    },
    "auxiliary_isolation_forest_signal": {
      "anomaly_count": 25,
      "training_scope": "Building 20 only (Auxiliary signal)",
      "warning": "Artifact twinergy_isoforest.pkl was trained exclusively on Building 20. Multi-building cross-inference is uncalibrated and preserved solely as auxiliary supporting evidence."
    },
    "anomalies": [
      {
        "timestamp": "2016-12-31T23:00:00",
        "actual_kwh": 175.42,
        "expected_kwh": 180.73,
        "residual": -5.31,
        "deviation_ratio": -0.029,
        "severity": "NORMAL",
        "is_anomaly": 0,
        "anomaly_direction": "NOMINAL",
        "anomaly_type": "NORMAL",
        "classification_reason": "Consumption is within normal range (relative deviation: -2.9%).",
        "isoforest_anomaly": 1,
        "isoforest_score": -0.12
      }
    ]
  }
  ```

---

### 3.6 Cost and Carbon Impact
- **Endpoint:** `GET /buildings/{building_id}/impact`
- **Query Parameters:**
  - `meter` *(optional, int, default: 0)*
  - `tariff_per_kwh` *(optional, float, default: 8.0)*: Tariff rate (₹/kWh)
  - `carbon_intensity_kg_per_kwh` *(optional, float, default: 0.70)*: Grid emissions factor (kg CO2e/kWh)
  - `cutoff_time` *(optional, str)*: Cutoff timestamp for forecast horizon
- **Purpose:** Converts the 24-hour forecasted energy trajectory into financial cost (₹) and carbon footprint (kg CO2e) metrics.
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings/0/impact?meter=0&tariff_per_kwh=8.0"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "meter": 0,
    "cutoff_time": "2016-12-30T23:00:00",
    "assumptions": {
      "tariff_per_kwh": 8.0,
      "carbon_intensity_kg_per_kwh": 0.70,
      "disclaimer": "Demonstration assumptions; not actual utility tariff or measured grid carbon intensity."
    },
    "summary": {
      "total_kwh": 4646.49,
      "total_cost": 37171.88,
      "total_carbon_kg": 3252.54,
      "average_hourly_cost": 1548.83,
      "average_hourly_carbon_kg": 135.52,
      "peak_cost_hour": "2016-12-31T00:00:00",
      "peak_carbon_hour": "2016-12-31T00:00:00"
    },
    "hourly_impact": [
      {
        "step": 1,
        "timestamp": "2016-12-31T00:00:00",
        "predicted_kwh": 230.06,
        "hourly_cost": 1840.46,
        "hourly_carbon_kg": 161.04
      }
    ]
  }
  ```

---

### 3.7 Energy Recommendations
- **Endpoint:** `GET /buildings/{building_id}/recommendations`
- **Query Parameters:**
  - `meter` *(optional, int, default: 0)*
  - `cost_threshold` *(optional, float, default: 10000.0)*: Custom cost alert threshold in ₹
  - `carbon_threshold` *(optional, float, default: 1000.0)*: Custom carbon alert threshold in kg CO2e
  - `cutoff_time` *(optional, str)*: Reference cutoff timestamp
- **Purpose:** Generates prioritized advisory recommendations (`CRITICAL` > `HIGH` > `MEDIUM` > `LOW` > `INFO`) with data-backed quantitative evidence from `TwinState`.
- **Example Request:**
  ```bash
  curl -s "http://127.0.0.1:8000/buildings/0/recommendations?meter=0"
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "meter": 0,
    "twin_state_status": "NORMAL",
    "total_count": 2,
    "generated_at": "2026-10-25T14:52:14Z",
    "recommendations": [
      {
        "recommendation_id": "REC-COST-001",
        "priority": "MEDIUM",
        "category": "COST",
        "title": "Consider Demand Management for High Cost Forecast",
        "action": "Evaluate load-shifting options to reduce peak-period energy charges.",
        "reason": "Projected 24-hour cost exceeds the configured cost optimization threshold.",
        "evidence": "Projected 24h cost is ₹37,171.88, exceeding the engineering assumption threshold of ₹10,000.00."
      },
      {
        "recommendation_id": "REC-CARB-002",
        "priority": "MEDIUM",
        "category": "CARBON",
        "title": "Review Carbon Emission Impact",
        "action": "Consider energy conservation measures during peak carbon intensity periods.",
        "reason": "Projected 24-hour carbon emissions exceed configured carbon target threshold.",
        "evidence": "Projected 24h carbon emissions are 3,252.50 kg CO2e, exceeding configured target threshold of 1,000.00 kg CO2e."
      }
    ]
  }
  ```

---

### 3.8 What-If Demand Simulation
- **Endpoint:** `POST /buildings/{building_id}/simulate`
- **Supported Scenario Types:**
  1. `ENERGY_MULTIPLIER`: Multiplies baseline demand by factor (e.g. `1.10` for +10% demand, `value > 0`).
  2. `ENERGY_REDUCTION`: Reduces baseline demand by fraction (e.g. `0.15` for 15% reduction, `0.0 <= value < 1.0`).
  3. `FIXED_KWH_ADDITION`: Adds fixed amount of kWh per hour (e.g. `10.0` for +10 kWh/hour, resulting in non-negative total).
- **Immutability Guarantee:** This simulation operates purely on the forecast array; it does **not** mutate historical records, retrain models, or overwrite the baseline `TwinState`.
- **Example Request:**
  ```bash
  curl -s -X POST "http://127.0.0.1:8000/buildings/0/simulate" \
    -H "Content-Type: application/json" \
    -d '{
      "scenario_type": "ENERGY_MULTIPLIER",
      "value": 1.10,
      "meter": 0
    }'
  ```
- **Example Response (200 OK):**
  ```json
  {
    "building_id": 0,
    "meter": 0,
    "scenario_type": "ENERGY_MULTIPLIER",
    "scenario_value": 1.10,
    "summary": {
      "baseline_total_kwh": 4646.49,
      "simulated_total_kwh": 5111.13,
      "delta_total_kwh": 464.65,
      "delta_total_percent": 10.0,
      "baseline_peak_kwh": 230.06,
      "simulated_peak_kwh": 253.06,
      "baseline_peak_hour": "2016-12-31T00:00:00",
      "simulated_peak_hour": "2016-12-31T00:00:00",
      "baseline_total_cost": 37171.88,
      "simulated_total_cost": 40889.07,
      "delta_total_cost": 3717.19,
      "baseline_total_carbon_kg": 3252.54,
      "simulated_total_carbon_kg": 3577.79,
      "delta_total_carbon_kg": 325.25,
      "tariff_per_kwh": 8.0,
      "carbon_intensity_kg_per_kwh": 0.70
    },
    "hourly_simulation": [
      {
        "timestamp": "2016-12-31T00:00:00",
        "baseline_kwh": 230.06,
        "simulated_kwh": 253.06,
        "delta_kwh": 23.01,
        "delta_percent": 10.0
      }
    ]
  }
  ```

---

## 4. HTTP Status Codes & Error Handling

The API adheres strictly to REST standards:

| Status Code | Reason | Description |
|---|---|---|
| `200 OK` | Success | Request was successful and data returned. |
| `400 Bad Request` | Logical Parameter Error | Scenario calculation leading to invalid state (e.g. negative energy) or forecast unavailable due to missing history/weather. |
| `404 Not Found` | Unknown Resource | Specified building ID does not exist, or building has no records for the requested meter ID. |
| `422 Unprocessable Content` | Schema / Validation Failure | Negative building ID/meter, out-of-bounds scenario values, or invalid timestamp strings. |
| `500 Internal Server Error` | Unexpected Server Error | Unhandled server error. Stack traces and local file paths are strictly suppressed in the client response. |

---

## 5. Known Limitations & Architectural Notes

1. **Historical Dataset vs. Physical Telemetry:**
   - The building `TwinState` representation is a software data structure constructed from historical observations in the ASHRAE GEPIII dataset, machine learning outputs, and impact calculators.
   - It is **not** an active physical digital twin connected to live IoT sensors, building automation systems, or physical actuators.
2. **Auxiliary Isolation Forest Training Scope:**
   - The auxiliary Isolation Forest model (`twinergy_isoforest.pkl`) was trained exclusively on Building 20. Cross-building inference is uncalibrated and preserved solely as secondary supporting context.
   - Primary anomaly detection relies strictly on the calibrated residual-based analyzer (`EnergyAnomalyAnalyzer`).
3. **Non-Causal Recommendations:**
   - Recommendations are rule-based advisory guidelines using verbs like *investigate*, *review*, and *consider*. They do not claim root-cause proof or automated HVAC control.
4. **Pure Simulations:**
   - What-if simulations apply transformations to pre-computed forecast horizons and do not retrain models. Baseline states are strictly immutable.
