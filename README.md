# 🏢 Twinergy
### AI-Driven Digital Twin for Building Energy Forecasting & Anomaly Detection

> A cloud-deployable digital twin that monitors building energy consumption in real time, forecasts the next 24 hours of usage, detects anomalies automatically, and explains *why* consumption is high - all without physical hardware.

---

## What Is This?

Twinergy is a software replica (digital twin) of a real building's energy system. It ingests historical IoT sensor data, learns consumption patterns using machine learning, and surfaces everything through a live interactive dashboard.

**Three things it does:**
- **Forecasts** energy consumption for the next 24 hours (LightGBM)
- **Detects anomalies** - unusual spikes or drops flagged in real time (Isolation Forest)
- **Explains predictions** - tells you *why* consumption is high right now (SHAP)

---

## Project Structure

```
twinergy/
├── docs/                        # Project documentation
|   └── Technical_Reference_doc  # Technical documentation
├── notebooks/
│   └── phase1.ipynb             # Full pipeline: EDA → cleaning → features → models
├── app.py                       # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── twinergy_lgbm.pkl            # Trained LightGBM forecasting model
├── twinergy_isoforest.pkl       # Trained Isolation Forest anomaly model
└── .gitignore                   # twinergy_clean.csv excluded (too large for git)
```

> `twinergy_clean.csv` is not tracked in git (500MB+). See setup instructions below to generate it.

---

## Dataset

**ASHRAE Great Energy Predictor III** - available free on Kaggle.

| Detail | Value |
|---|---|
| Total rows | 20,216,100 readings |
| Buildings | 1,449 real buildings |
| Sites | 16 locations worldwide |
| Duration | Full year (Jan–Dec 2016) |
| Meter types | Electricity, Chilled Water, Steam, Hot Water |
| Extra | Hourly weather data per site |

Download: https://www.kaggle.com/competitions/ashrae-energy-prediction/data

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | `pandas`, `numpy` |
| Forecasting model | `LightGBM` |
| Anomaly detection | `Isolation Forest` (scikit-learn) |
| Explainability | `SHAP` |
| Dashboard | `Streamlit` |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |

---

## Setup - Run This on Any Device

### Prerequisites
- Python 3.9 or higher
- Git
- ~3GB free disk space (for the dataset + models)

### Step 1 - Clone the repo

```bash
git clone https://github.com/your-username/twinergy.git
cd twinergy
```

### Step 2 - Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 - Get the dataset

1. Go to https://www.kaggle.com/competitions/ashrae-energy-prediction/data
2. Accept the competition rules and download the zip
3. Extract only these three files into a folder called `ashrae/`:
   - `train.csv`
   - `building_metadata.csv`
   - `weather_train.csv`

### Step 4 - Run the notebook to generate models + clean data

Open `notebooks/phase1.ipynb` in Kaggle Notebooks (recommended - 30GB RAM) or Jupyter.

If using **Kaggle Notebooks**:
- Add the ASHRAE dataset via Add Data → search `ashrae-energy-prediction`
- Run all cells top to bottom
- Download the outputs from the Output tab:
  - `twinergy_clean.csv`
  - `twinergy_lgbm.pkl`
  - `twinergy_isoforest.pkl`
- Place all three in the root `twinergy/` folder

If using **local Jupyter**:
- Update the file paths in Step 1 of the notebook to point to your `ashrae/` folder
- Run all cells
- Outputs will be saved to the same directory as the notebook

### Step 5 - Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Results

| Metric | Value |
|---|---|
| RMSE (log scale) | 0.1216 |
| MAPE | 13.09% |
| Accuracy proxy |86.91% |
| Training data | Jan-Oct 2016 |
| Test data | Nov-Dec 2016 |

---

## Dashboard Panels

| Panel | Description |
|---|---|
| Live Overview | Current vs predicted consumption with delta |
| 24hr Forecast | LightGBM prediction with confidence band |
| Anomaly Alerts | Timestamped alerts with severity (Low / Medium / High) |
| SHAP Explainer | Feature importance - why is consumption high right now? |
| Monthly Overview | Full year consumption trend with peak overlay |

---

## Research Papers

| # | Paper | Venue |
|---|---|---|
| 1 ⭐ | Explainable AI for Energy Prediction and Anomaly Detection in Smart Buildings | ACM BuildSys '23 |
| 2 | Limitations of ML for Building Energy Prediction: ASHRAE GEPIII Error Analysis | NUS 2021 |
| 3 | Anomaly Detection Framework for Digital Twin Driven Cyber-Physical Systems | ACM ICCPS '21 |
| 4 | Dynamic Urban Digital Twin for Microclimate Studies | ACM BuildSys '23 |
| 5 | eptk: Open-Source Energy Prediction Toolkit | ACM BuildSys '22 |

---

## Team Members

[A Soundarya Lahari](https://github.com/sounds034)

[Bharath Kumar B D](https://github.com/BharathKumarBD)

[Gauri S](https://github.com/gauris8)

[Gautham KV](https://github.com/Gauthamkv14)