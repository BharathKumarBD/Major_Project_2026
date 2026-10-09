# 🏢 Twinergy 2.0
### AI-Driven Digital Twin for Building Energy Forecasting, Anomaly Diagnosis & Simulation

> A software digital twin replica of building energy systems that monitors energy consumption, provides recursive 24-hour forward forecasts without future target leakage, diagnoses anomalies via contextual residual analysis, computes local TreeSHAP attributions, forecasts cost and carbon emissions, aggregates building Twin State, and runs deterministic what-if scenario simulations.

---

## 1. System Architecture & Verified Data Flow

```
                         ASHRAE GEPIII Dataset
                                  │
                                  ▼
                    Data Pipeline & Cleaning
              (src/data/preprocessing.py & feature_engineering.py)
                                  │
                                  ▼
                   Optimized Parquet / DuckDB
                 (twinergy_clean.parquet & DataLoader)
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
   24h Forecaster          Anomaly Engine          Model Explainer
 (LightGBM Recursive)    (Residual & Category)    (Local TreeSHAP)
   [src/models/            [src/models/            [src/models/
    forecasting.py]         anomaly.py]             explainability.py]
          │                       │                       │
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
                         Impact Calculator
                      (Cost & Carbon Metrics)
                       [src/models/impact.py]
                                  │
                                  ▼
                         Building Twin State
                     (Software Aggregation Layer)
                      [src/models/twin_state.py]
                                  │
                                  ▼
                       What-If Scenario Engine
                    (Simulated Demand & Impacts)
                      [src/models/simulation.py]
                                  │
                                  ▼
                     Interactive Streamlit App
                             [app.py]
```

---

## 2. Implemented Capabilities (Steps 0 through 9)

### ✅ Step 0 & 1 — Data Foundation & Storage
- **Reproducible Data Foundation:** Modular cleaning and feature engineering in [`src/data/preprocessing.py`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/data/preprocessing.py) and [`src/data/feature_engineering.py`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/data/feature_engineering.py).
- **High-Performance Parquet & DuckDB Querying:** Automated conversion script in [`scripts/prepare_data.py`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/scripts/prepare_data.py) yielding `twinergy_clean.parquet` (6.6x compression vs CSV) with sub-second out-of-core pushdown queries via [`src/data/loader.py`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/data/loader.py).
- **Explicit Data Contract:** Full column dictionaries, dtypes, missing value policies, and proxy definitions in [`DATA_CONTRACT.md`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/DATA_CONTRACT.md).
- **Automated Validation Suite:** 9 automated contract validation checks in [`src/data/validation.py`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/data/validation.py).

### ✅ Step 2 & 3 — Core ML Engine & 24-Hour Forward Forecasting
- **Extracted Model Services:** Decoupled forecasting and anomaly services in `src/models/`.
- **True Recursive Multi-Step 24h Forecaster:** [`EnergyForecaster.forecast_24h`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/models/forecasting.py) dynamically rolls predictions $t+1 \dots t+24$ through the autoregressive lag and rolling buffers without future ground-truth target leakage.
- **Strict Leakage Safety:** Comprehensive unit tests verify that future ground-truth energy values are never used during multi-step inference.

### ✅ Step 4 & 6 — Contextual Anomaly Detection & Deterministic Classification
- **Primary Residual Engine:** Evaluates observed consumption against model-expected baselines ($e_t = y_t - \hat{y}_t$, $\delta_t = e_t / \max(\hat{y}_t, \epsilon)$) with three severity tiers (`NORMAL`, `WARNING`, `CRITICAL`).
- **Deterministic Anomaly Classification:** Classifies deviations into 5 interpretable categories: `NORMAL`, `LOW_DEMAND`, `RAPID_SPIKE`, `PERSISTENT_HIGH`, and `HIGH_DEMAND` with human-readable diagnostic evidence.
- **Single-Building Scope Isolation:** The legacy `twinergy_isoforest.pkl` model artifact is preserved strictly as secondary auxiliary evidence with an explicit single-building (`Building 20`) training scope disclosure.

### ✅ Step 5 — Local TreeSHAP Explainability
- **Sample-Bounded Local TreeSHAP:** [`ModelExplainer.explain_instance`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/models/explainability.py) computes exact additive Shapley attributions: $\hat{y} = E[\hat{y}] + \sum \phi_i$.
- **Clear Disclaimers:** Explicitly distinguishes global tree gain split importance from local additive SHAP attribution, noting that attribution reflects model feature weighting rather than physical real-world causality.

### ✅ Step 7 — Cost & Carbon Impact Forecasting
- **Impact Service Layer:** [`EnergyImpactCalculator`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/models/impact.py) maps 24-hour forecasted consumption to hourly and cumulative cost (₹) and carbon emissions ($\text{kg CO}_2\text{e}$).
- **Configurable Parameters:** Default baseline tariff (₹8.0/kWh) and grid carbon factor ($0.70\text{ kg CO}_2\text{e/kWh}$) with strict non-negative and finite value validation.

### ✅ Step 8 — Centralized Building Twin State
- **Software Digital Twin Model:** [`TwinState`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/models/twin_state.py) dataclass aggregating building identity, current energy metrics, residual anomaly diagnosis, 24-hour forecast trajectory, cost/carbon impacts, and deterministic status (`NORMAL`, `WARNING`, `CRITICAL`, `UNAVAILABLE`).
- **Serialization & Immutability:** Full JSON serialization via `.to_dict()` and timezone-aware UTC metadata timestamps.

### ✅ Step 9 — What-If Energy Scenario Simulation
- **Deterministic Scenario Engine:** [`EnergySimulationEngine`](file:///c:/Users/User/OneDrive/Desktop/Major%20Project/phase_1/twinenergy/src/models/simulation.py) evaluates controlled scenario modifications on forecast trajectories:
  1. `ENERGY_MULTIPLIER` (e.g. +10% demand surge)
  2. `ENERGY_REDUCTION` (e.g. 10% load curtailment)
  3. `FIXED_KWH_ADDITION` (e.g. +5.0 kWh/h baseload addition)
  4. `OCCUPANCY_CHANGE` (e.g. +20% occupancy impact during business hours)
  5. `TEMPERATURE_CHANGE` (e.g. +3.0°C cooling demand sensitivity)
  6. `EFFICIENCY_MULTIPLIER` (e.g. 15% efficiency improvement)
- **Baseline vs Scenario Comparison:** Returns delta energy, delta percentages, cost deltas, carbon deltas, and peak demand hours without mutating original forecasts or TwinState.

---

## 3. Project Structure

```
twinenergy/
├── data/
│   └── sample/
│       └── twinergy_sample.parquet     # Lightweight development sample dataset (1.85 MB)
├── docs/
│   └── Technical_Reference_doc.docx    # Project background documentation
├── notebooks/
│   └── phase1.ipynb                    # Exploratory Phase 1 notebook
├── scripts/
│   └── prepare_data.py                 # Automated CSV to Parquet & sample generator
├── src/
│   ├── data/
│   │   ├── feature_engineering.py      # Lags, rollings, calendar, occupancy proxy
│   │   ├── loader.py                   # DuckDB/PyArrow analytical data loader
│   │   ├── preprocessing.py            # Weather cleaning, outlier filtering, merging
│   │   ├── schema.py                   # Canonical 37-column schema & 22-feature contracts
│   │   └── validation.py               # Automated schema & data contract test suite
│   └── models/
│       ├── anomaly.py                  # Residual anomaly analyzer & rule classifier
│       ├── explainability.py           # Local TreeSHAP & global tree gain explainer
│       ├── forecasting.py              # Recursive 24-hour forward forecaster
│       ├── impact.py                   # Electricity cost & carbon emissions calculator
│       ├── simulation.py               # What-if scenario simulation engine
│       └── twin_state.py               # TwinState data model and builder service
├── tests/
│   ├── test_anomaly.py                 # Anomaly residual foundation tests
│   ├── test_anomaly_classification.py  # Anomaly category & rule priority tests
│   ├── test_data_foundation.py         # Loader, schema, and model contract tests
│   ├── test_explainability.py          # Local SHAP structure and additive property tests
│   ├── test_forecast_24h.py            # 24h recursive forecasting & zero-leakage tests
│   ├── test_impact.py                  # Cost & carbon calculation and validation tests
│   ├── test_models.py                  # Model loading and inference regression tests
│   ├── test_simulation.py              # What-if simulation scenarios and immutability tests
│   └── test_twin_state.py              # TwinState creation, status, and serialization tests
├── app.py                              # Streamlit live interactive digital twin dashboard
├── DATA_CONTRACT.md                    # Data contract & feature specification
├── README.md                           # System architecture & verification guide
├── requirements.txt                    # Python dependencies
├── twinergy_clean.parquet              # Columnar Parquet dataset (916.7 MB)
├── twinergy_isoforest.pkl              # Trained Isolation Forest model artifact
└── twinergy_lgbm.pkl                   # Trained LightGBM forecasting model artifact
```

---

## 4. Setup & Execution Instructions

### Prerequisites
- Python 3.9 to 3.13
- Git

### Step 1 — Clone and Install Dependencies

```bash
git clone https://github.com/your-username/twinenergy.git
cd twinenergy
pip install -r requirements.txt
```

### Step 2 — Verify or Prepare Data Foundation

If `twinergy_clean.parquet` is not yet generated from raw/clean CSV:

```bash
python scripts/prepare_data.py
```

To run data contract validation:

```bash
python src/data/validation.py
```

### Step 3 — Run the Test Suite

Execute all 63 unit and regression tests:

```bash
pytest
```

### Step 4 — Launch the Digital Twin Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 5. Critical Domain Disclosures & Limitations

1. **Software Digital Twin Replica:** TwinState is a software representation computed from historical ASHRAE GEPIII time-series data, ML forecasts, and impact models. It is not connected to physical building actuators or live IoT telemetry.
2. **Occupancy Proxy (`occupancy_proxy`):** This feature is strictly an engineered mathematical heuristic ($\text{is\_business\_hours} \times \text{rolling\_mean\_6hr}$), not sensor-measured room headcount.
3. **Model Attribution vs Causality:** Local SHAP explanations represent mathematical model feature weighting for a LightGBM prediction and do not prove physical causality in the building.
4. **Tariff and Carbon Assumptions:** Cost (₹8.0/kWh) and carbon ($0.70\text{ kg CO}_2\text{e/kWh}$) are configurable demonstration parameters, not measured real-time grid emissions.
5. **Isolation Forest Training Scope:** The artifact `twinergy_isoforest.pkl` was trained exclusively on `Building 20`. Multi-building anomaly detection is performed via the contextual residual engine (`EnergyAnomalyAnalyzer`).

---

## 6. Planned Subsequent Stages (Future Scope)

The following capabilities are scheduled for subsequent development steps per the project roadmap:
- **Step 10:** Simulation-Backed Recommendation Engine
- **Step 11:** FastAPI Modular Monolith Backend
- **Step 12:** React + TypeScript + Vite Frontend
- **Step 13:** Guarded Natural Language AI Assistant
- **Step 14:** Self-Adaptive Drift Detection & Model Recalibration Tracking
- **Step 15:** Historical Telemetry Replay Engine