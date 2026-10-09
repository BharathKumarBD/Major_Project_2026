import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import DataLoader
from src.data.schema import LIGHTGBM_FEATURES, ANOMALY_FEATURES
from src.models.forecasting import load_forecaster
from src.models.anomaly import load_anomaly_detector, load_anomaly_analyzer
from src.models.explainability import load_explainer
from src.models.impact import EnergyImpactCalculator, load_impact_calculator
from src.models.twin_state import TwinStateBuilder, build_twin_state
from src.models.simulation import EnergySimulationEngine, load_simulation_engine
from src.models.recommendations import RecommendationEngine, load_recommendation_engine

FEATURES = LIGHTGBM_FEATURES

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Twinergy",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2563eb;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #93c5fd;
        margin-top: 4px;
    }
    .alert-high {
        background: #450a0a;
        border-left: 4px solid #dc2626;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #fca5a5;
    }
    .alert-medium {
        background: #431407;
        border-left: 4px solid #f97316;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #fdba74;
    }
    .alert-low {
        background: #052e16;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        color: #86efac;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e3a5f;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: #93c5fd !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD SERVICES & DATA ACCESS
# ─────────────────────────────────────────
@st.cache_resource
def get_data_loader():
    return DataLoader()

@st.cache_resource
def get_forecaster():
    return load_forecaster()

@st.cache_resource
def get_anomaly_analyzer():
    return load_anomaly_analyzer(forecaster=get_forecaster())

@st.cache_resource
def get_anomaly_detector():
    return load_anomaly_detector(forecaster=get_forecaster())

@st.cache_resource
def get_explainer():
    return load_explainer(forecaster=get_forecaster())

@st.cache_resource
def get_impact_calculator():
    return EnergyImpactCalculator()

@st.cache_resource
def get_twin_state_builder():
    return TwinStateBuilder()

@st.cache_resource
def get_simulation_engine():
    return EnergySimulationEngine()

@st.cache_resource
def get_recommendation_engine():
    return RecommendationEngine()

loader             = get_data_loader()
forecaster         = get_forecaster()
analyzer           = get_anomaly_analyzer()
detector           = get_anomaly_detector()
explainer          = get_explainer()
impact_calc        = get_impact_calculator()
twin_state_builder = get_twin_state_builder()
sim_engine         = get_simulation_engine()
rec_engine         = get_recommendation_engine()

# ─────────────────────────────────────────
# ─────────────────────────────────────────
# DATA ACCESS HELPER
# ─────────────────────────────────────────
@st.cache_data
def get_building_data(b_id: int):
    return loader.load(building_id=b_id, meter=0)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Twinergy")
    st.markdown("*AI-Driven Digital Twin*")
    st.markdown("---")

    building_ids = loader.get_building_ids(meter=0)
    selected_building = st.selectbox("Select Building", building_ids, index=0)

    st.markdown("---")

    b_meta      = loader.get_building_metadata(selected_building)
    primary_use = b_meta['primary_use']
    sq_ft       = b_meta['square_feet']
    site        = b_meta['site_id']

    st.markdown(f"**Primary Use:** {primary_use}")
    st.markdown(f"**Size:** {sq_ft:,.0f} sq ft")
    st.markdown(f"**Site ID:** {site}")

    st.markdown("---")
    st.markdown("**View Window**")
    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    month_names_list = list(month_map.keys())

    # Pre-fetch building data to determine available observation months
    b_df = get_building_data(selected_building)
    available_month_nums = sorted(b_df['timestamp'].dt.month.unique().tolist()) if len(b_df) > 0 else []

    # Default to May (month 5) if available, otherwise first available month
    if 5 in available_month_nums:
        default_idx = 4
    elif len(available_month_nums) > 0:
        default_idx = available_month_nums[0] - 1
    else:
        default_idx = 0

    selected_month = st.selectbox("Month", month_names_list, index=default_idx)
    month_num = month_map[selected_month]

    if month_num not in available_month_nums:
        avail_names = [month_names_list[m - 1] for m in available_month_nums]
        st.warning(f"⚠️ No recorded observations for {selected_month} (Available: {avail_names[0]}–{avail_names[-1]})" if len(avail_names) >= 2 else "⚠️ No recorded observations for this month.")

    st.markdown("---")
    st.caption("ASHRAE GEPIII Dataset")
    st.caption("LightGBM + Isolation Forest")
    st.caption("MAPE: 13.09% | RMSE: 0.1216")
    st.caption("⚠️ Isolation Forest baseline trained on Building 20 (single-building artifact)")

# ─────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────
b_month = b_df[b_df['timestamp'].dt.month == month_num].copy()

# Predictions (via EnergyForecaster)
b_month_clean = b_month.dropna(subset=FEATURES)
if len(b_month_clean) > 0:
    b_month_clean = b_month_clean.copy()
    b_month_clean['predicted'] = forecaster.predict(b_month_clean, return_kwh=True)
    b_month_clean['actual']    = np.expm1(b_month_clean['meter_reading_log'])

# Anomaly detection (via EnergyAnomalyAnalyzer & AnomalyDetector)
try:
    if len(b_month) > 0:
        b_month_anom = detector.detect(b_month)
        anomaly_events = b_month_anom[b_month_anom['is_anomaly'] == 1].copy()
    else:
        b_month_anom = pd.DataFrame()
        anomaly_events = pd.DataFrame()
except Exception as exc:
    b_month_anom = pd.DataFrame()
    anomaly_events = pd.DataFrame()
    st.warning(f"⚠️ **Anomaly evaluation unavailable for Building {selected_building}:** {exc}")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# 🏢 Twinergy")
st.markdown(f"**AI-Driven Digital Twin** — Building `{selected_building}` · {primary_use} · {selected_month} 2016")
st.markdown("---")

# ─────────────────────────────────────────
# PANEL 1 — KPI METRICS
# ─────────────────────────────────────────
if len(b_month_clean) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)
    current_kwh   = b_month_clean['actual'].iloc[-1]
    predicted_kwh = b_month_clean['predicted'].iloc[-1]
    avg_kwh       = b_month_clean['actual'].mean()
    total_kwh     = b_month_clean['actual'].sum()
    n_anomalies   = len(anomaly_events)
    carbon_factor = impact_calc.carbon_intensity_kg_per_kwh
    co2_kg        = total_kwh * carbon_factor  # Configured carbon intensity (kg CO2e / kWh)

    delta = ((current_kwh - predicted_kwh) / predicted_kwh) * 100
    delta_str = f"{'▲' if delta > 0 else '▼'} {abs(delta):.1f}% vs predicted"

    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{current_kwh:.0f}</div>
            <div class='metric-label'>Current kWh</div>
            <div style='color:#fbbf24;font-size:0.75rem;margin-top:4px'>{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{predicted_kwh:.0f}</div>
            <div class='metric-label'>Predicted kWh</div>
            <div style='color:#86efac;font-size:0.75rem;margin-top:4px'>LightGBM forecast</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_kwh:.0f}</div>
            <div class='metric-label'>Monthly Avg kWh</div>
            <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>{selected_month} 2016</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        alert_color = '#dc2626' if n_anomalies > 10 else '#f97316' if n_anomalies > 3 else '#16a34a'
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{alert_color}'>{n_anomalies}</div>
            <div class='metric-label'>Anomalies Detected</div>
            <div style='color:{alert_color};font-size:0.75rem;margin-top:4px'>Residual Engine (|dev| > 25%)</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{co2_kg:,.0f}</div>
            <div class='metric-label'>CO₂ Estimate (kg)</div>
            <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>@{carbon_factor:.2f} kg/kWh</div>
        </div>""", unsafe_allow_html=True)
else:
    avail_names = [month_names_list[m - 1] for m in available_month_nums]
    avail_str = f"{avail_names[0]} through {avail_names[-1]}" if len(avail_names) >= 2 else (", ".join(avail_names) if avail_names else "None")
    st.info(
        f"ℹ️ **No Historical Observations for {selected_month} 2016:** Building `{selected_building}` has no recorded meter readings for {selected_month} in the ASHRAE GEPIII dataset. "
        f"Available observation window for this building: **{avail_str}**."
    )

st.markdown("---")

# ─────────────────────────────────────────
# PANEL 2 — FORECAST CHART
# ─────────────────────────────────────────
st.markdown("<div class='section-title'>📈 Energy Forecast — Actual vs Predicted</div>", unsafe_allow_html=True)

if len(b_month_clean) > 0:
    fig = go.Figure()

    # Illustrative deviation band (+/- 15%)
    fig.add_trace(go.Scatter(
        x=pd.concat([b_month_clean['timestamp'], b_month_clean['timestamp'][::-1]]),
        y=pd.concat([b_month_clean['predicted'] * 1.15, (b_month_clean['predicted'] * 0.85)[::-1]]),
        fill='toself', fillcolor='rgba(37,99,235,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Nominal Range (±15%)', hoverinfo='skip'
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=b_month_clean['timestamp'], y=b_month_clean['predicted'],
        mode='lines', name='Predicted',
        line=dict(color='#3b82f6', width=2, dash='dash')
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=b_month_clean['timestamp'], y=b_month_clean['actual'],
        mode='lines', name='Actual',
        line=dict(color='#f97316', width=1.8)
    ))

    # Anomaly dots on forecast chart
    if len(anomaly_events) > 0:
        anom_merged = anomaly_events.merge(
            b_month_clean[['timestamp','actual']], on='timestamp', how='left'
        )
        fig.add_trace(go.Scatter(
            x=anom_merged['timestamp'],
            y=np.expm1(anom_merged['meter_reading_log']),
            mode='markers', name='Anomaly',
            marker=dict(color='#dc2626', size=8, symbol='x')
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f', title='kWh')
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        f"ℹ️ **Forecast Chart Unavailable for {selected_month} 2016:** No historical meter observations exist for Building `{selected_building}` in {selected_month} to compare actual vs predicted demand."
    )

# ─────────────────────────────────────────
# PANEL 2B — 24-HOUR ENERGY FORECAST
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🔮 24-Hour Energy Forecast</div>", unsafe_allow_html=True)
st.caption("Recursive 24-hour multi-step energy forecast starting strictly after the latest historical observation. Future lag and rolling features are recursively computed from prior step predictions without target leakage.")

@st.cache_data
def get_building_forecast_24h(b_id: int):
    return forecaster.forecast_24h(building_id=b_id, meter=0)

try:
    forecast_24h_df = get_building_forecast_24h(selected_building)
except Exception as exc:
    forecast_24h_df = None
    st.warning(
        f"⚠️ **24-Hour Energy Forecast Unavailable for Building {selected_building}:** {exc}"
    )

if forecast_24h_df is not None and len(forecast_24h_df) > 0:
    total_24h_kwh = forecast_24h_df["predicted_kwh"].sum()
    peak_24h_kwh  = forecast_24h_df["predicted_kwh"].max()
    min_24h_kwh   = forecast_24h_df["predicted_kwh"].min()
    peak_idx      = forecast_24h_df["predicted_kwh"].idxmax()
    min_idx       = forecast_24h_df["predicted_kwh"].idxmin()
    peak_ts       = forecast_24h_df.loc[peak_idx, "timestamp"]
    min_ts        = forecast_24h_df.loc[min_idx, "timestamp"]
    weather_src   = forecast_24h_df["weather_source"].iloc[0]
    first_ts      = forecast_24h_df["timestamp"].iloc[0]
    last_ts       = forecast_24h_df["timestamp"].iloc[-1]

    # Forecast KPI metrics
    fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

    with fc_col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{total_24h_kwh:,.1f}</div>
            <div class='metric-label'>Total Forecast (24h kWh)</div>
            <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>24-Hour Cumulative Sum</div>
        </div>""", unsafe_allow_html=True)

    with fc_col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#fbbf24'>{peak_24h_kwh:.1f}</div>
            <div class='metric-label'>Peak Predicted Hourly Demand</div>
            <div style='color:#fbbf24;font-size:0.75rem;margin-top:4px'>at {peak_ts.strftime('%H:%M (%d %b)')}</div>
        </div>""", unsafe_allow_html=True)

    with fc_col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#86efac'>{min_24h_kwh:.1f}</div>
            <div class='metric-label'>Minimum Predicted Demand</div>
            <div style='color:#86efac;font-size:0.75rem;margin-top:4px'>at {min_ts.strftime('%H:%M (%d %b)')}</div>
        </div>""", unsafe_allow_html=True)

    with fc_col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='font-size:1.35rem;padding-top:6px'>{weather_src}</div>
            <div class='metric-label'>Weather Assumption</div>
            <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>{first_ts.strftime('%d %b %H:%M')} – {last_ts.strftime('%d %b %H:%M')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Line Chart & Table side-by-side
    chart_col, table_col = st.columns([1.35, 1.0])

    with chart_col:
        st.caption("🔮 **Forecast (Predicted kWh)** — Hourly multi-step trajectory")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=forecast_24h_df["timestamp"],
            y=forecast_24h_df["predicted_kwh"],
            mode="lines+markers",
            name="Forecast (Predicted kWh)",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#0284c7"),
            hovertemplate="<b>Forecast Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Predicted Demand:</b> %{y:.2f} kWh<extra></extra>",
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            height=320,
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis=dict(title="Forecast Timestamp", gridcolor="#1e3a5f"),
            yaxis=dict(title="Predicted kWh", gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with table_col:
        st.caption("📋 **Hourly Forecast Values** (Next 24 Hours)")
        display_df = pd.DataFrame({
            "Step": forecast_24h_df["step"],
            "Forecast Timestamp": forecast_24h_df["timestamp"].dt.strftime("%d %b %H:%M"),
            "Predicted kWh": forecast_24h_df["predicted_kwh"].round(2),
            "Air Temp (°C)": forecast_24h_df["air_temperature"].round(1),
        })
        st.dataframe(display_df, use_container_width=True, height=295, hide_index=True)

# ─────────────────────────────────────────
# PANEL 2C — COST & CARBON IMPACT
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>💰 Cost & Carbon Impact</div>", unsafe_allow_html=True)

impact = None
if forecast_24h_df is not None and len(forecast_24h_df) > 0:
    try:
        impact = impact_calc.calculate_impact(forecast_24h_df)

        # Assumptions note using calculator values
        tariff_val = impact["tariff_per_kwh"]
        carbon_val = impact["carbon_intensity_kg_per_kwh"]
        st.info(
            f"Cost uses a configurable tariff of ₹{tariff_val:.2f}/kWh.\n\n"
            f"Carbon uses a configurable factor of {carbon_val:.2f} kg CO₂e/kWh.\n\n"
            "These are demonstration assumptions and are not the building's actual utility tariff or measured grid carbon intensity."
        )

        # Peak timestamps directly from calculator
        peak_cost_hour = impact["peak_cost_hour"]
        peak_carbon_hour = impact["peak_carbon_hour"]
        peak_cost_ts = peak_cost_hour.strftime("%H:%M (%d %b)") if peak_cost_hour is not None else "N/A"
        peak_carbon_ts = peak_carbon_hour.strftime("%H:%M (%d %b)") if peak_carbon_hour is not None else "N/A"

        # Summary metrics
        imp_col1, imp_col2, imp_col3, imp_col4, imp_col5 = st.columns(5)

        with imp_col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{impact['total_kwh']:,.2f} kWh</div>
                <div class='metric-label'>24h Energy</div>
                <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>Forecast Horizon</div>
            </div>""", unsafe_allow_html=True)

        with imp_col2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#fbbf24'>₹{impact['total_cost']:,.2f}</div>
                <div class='metric-label'>24h Cost</div>
                <div style='color:#fbbf24;font-size:0.75rem;margin-top:4px'>@ ₹{tariff_val:.2f}/kWh</div>
            </div>""", unsafe_allow_html=True)

        with imp_col3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#34d399'>{impact['total_carbon_kg']:,.2f} kg CO₂e</div>
                <div class='metric-label'>24h Carbon Emissions</div>
                <div style='color:#34d399;font-size:0.75rem;margin-top:4px'>@ {carbon_val:.2f} kg/kWh</div>
            </div>""", unsafe_allow_html=True)

        with imp_col4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='font-size:1.35rem;padding-top:6px'>₹{impact['average_hourly_cost']:,.2f}</div>
                <div class='metric-label'>Average Hourly Cost</div>
                <div style='color:#fbbf24;font-size:0.75rem;margin-top:4px'>Peak: {peak_cost_ts}</div>
            </div>""", unsafe_allow_html=True)

        with imp_col5:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='font-size:1.35rem;padding-top:6px'>{impact['average_hourly_carbon_kg']:,.2f} kg CO₂e</div>
                <div class='metric-label'>Average Hourly Carbon</div>
                <div style='color:#34d399;font-size:0.75rem;margin-top:4px'>Peak: {peak_carbon_ts}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"📌 **Peak Impact Hours:** &nbsp;&nbsp;"
            f"**Peak Cost Hour:** `{peak_cost_ts}` &nbsp;|&nbsp; "
            f"**Peak Carbon Hour:** `{peak_carbon_ts}`"
        )
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Hourly chart & table
        imp_chart_col, imp_table_col = st.columns([1.35, 1.0])
        hourly_imp_df = impact["hourly_df"]

        with imp_chart_col:
            st.caption("📊 **Hourly Impact Trajectory** (Predicted Energy vs Cost & Carbon)")
            from plotly.subplots import make_subplots
            fig_imp = make_subplots(specs=[[{"secondary_y": True}]])

            # Primary Y-axis: Predicted Energy (kWh)
            fig_imp.add_trace(
                go.Scatter(
                    x=hourly_imp_df["timestamp"],
                    y=hourly_imp_df["predicted_kwh"],
                    mode="lines+markers",
                    name="Energy (kWh)",
                    line=dict(color="#38bdf8", width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Predicted Energy:</b> %{y:.2f} kWh<extra></extra>",
                ),
                secondary_y=False,
            )

            # Secondary Y-axis: Hourly Cost (₹)
            fig_imp.add_trace(
                go.Scatter(
                    x=hourly_imp_df["timestamp"],
                    y=hourly_imp_df["hourly_cost"],
                    mode="lines",
                    name="Cost (₹)",
                    line=dict(color="#fbbf24", width=2, dash="dash"),
                    hovertemplate="<b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Hourly Cost:</b> ₹%{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

            # Secondary Y-axis: Hourly Carbon (kg CO₂e)
            fig_imp.add_trace(
                go.Scatter(
                    x=hourly_imp_df["timestamp"],
                    y=hourly_imp_df["hourly_carbon_kg"],
                    mode="lines",
                    name="Carbon (kg CO₂e)",
                    line=dict(color="#34d399", width=2, dash="dot"),
                    hovertemplate="<b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Hourly Carbon:</b> %{y:.2f} kg CO₂e<extra></extra>",
                ),
                secondary_y=True,
            )

            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                height=320,
                margin=dict(l=10, r=10, t=15, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Timestamp", gridcolor="#1e3a5f"),
            )
            fig_imp.update_yaxes(title_text="Predicted Energy (kWh)", gridcolor="#1e3a5f", secondary_y=False)
            fig_imp.update_yaxes(title_text="Cost (₹) / Carbon (kg CO₂e)", gridcolor="#1e3a5f", secondary_y=True)

            st.plotly_chart(fig_imp, use_container_width=True)

        with imp_table_col:
            st.caption("📋 **Hourly Cost & Carbon Forecast**")
            display_imp_df = pd.DataFrame({
                "Timestamp": hourly_imp_df["timestamp"].dt.strftime("%d %b %H:%M"),
                "Predicted Energy (kWh)": hourly_imp_df["predicted_kwh"].round(2),
                "Cost (₹)": hourly_imp_df["hourly_cost"].round(2),
                "Carbon (kg CO₂e)": hourly_imp_df["hourly_carbon_kg"].round(2),
            })
            st.dataframe(display_imp_df, use_container_width=True, height=295, hide_index=True)

    except Exception as exc:
        st.warning(f"⚠️ **Cost & Carbon Impact Forecast Unavailable:** A valid 24-hour energy forecast is required ({exc}).")
else:
    st.warning("⚠️ **Cost & Carbon Impact Forecast Unavailable:** A valid 24-hour energy forecast is required.")

# ─────────────────────────────────────────
# PANEL 2D — BUILDING TWIN STATE
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🏢 Building Twin State</div>", unsafe_allow_html=True)
st.caption(
    "Current software representation of the selected building's energy state based on the available "
    "historical data, forecasts, anomaly analysis, and impact calculations."
)

# Reference timestamp for TwinState
if len(b_month_clean) > 0:
    ref_ts = b_month_clean['timestamp'].iloc[-1]
elif len(b_month) > 0:
    ref_ts = b_month['timestamp'].max()
else:
    ref_ts = pd.Timestamp.now()

# Build TwinState using existing computed service outputs
try:
    twin_state = twin_state_builder.build(
        building_id=selected_building,
        timestamp=ref_ts,
        meter=0,
        anomaly_analysis=b_month_anom if len(b_month_anom) > 0 else None,
        forecast_df=forecast_24h_df if (forecast_24h_df is not None and len(forecast_24h_df) > 0) else None,
        impact_dict=impact if impact is not None else None,
    )
except Exception as exc:
    st.warning(f"⚠️ **Building Twin State Construction Error:** {exc}")
    twin_state = None

if twin_state is not None:
    # 1. State Status Banner
    status = twin_state.state_status
    if status == "NORMAL":
        status_color = "#16a34a"
        status_bg = "#052e16"
        status_icon = "🟢"
    elif status == "WARNING":
        status_color = "#f97316"
        status_bg = "#431407"
        status_icon = "🟡"
    elif status == "CRITICAL":
        status_color = "#dc2626"
        status_bg = "#450a0a"
        status_icon = "🔴"
    else:
        status_color = "#94a3b8"
        status_bg = "#1e293b"
        status_icon = "⚪"

    ref_ts_fmt = twin_state.timestamp.strftime("%Y-%m-%d %H:%M") if twin_state.timestamp is not None else "N/A"
    gen_at_fmt = twin_state.generated_at if twin_state.generated_at else "N/A"

    st.markdown(
        f"""<div style='background-color:{status_bg};border-left:4px solid {status_color};padding:14px;border-radius:8px;margin-bottom:16px;'>
            <div style='font-size:1.15rem;font-weight:700;color:{status_color};'>
                {status_icon} Twin State Status: {status}
            </div>
            <div style='font-size:0.85rem;color:#cbd5e1;margin-top:6px;'>
                <b>Reference Timestamp (Software State):</b> {ref_ts_fmt} &nbsp;|&nbsp; 
                <b>State Generated At:</b> {gen_at_fmt}
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    # 2. State Overview Columns
    ts_col1, ts_col2, ts_col3 = st.columns(3)

    with ts_col1:
        st.markdown("#### ⚡ Current Energy State")
        act_fmt = f"{twin_state.actual_kwh:.2f} kWh" if twin_state.actual_kwh is not None else "N/A"
        exp_fmt = f"{twin_state.expected_kwh:.2f} kWh" if twin_state.expected_kwh is not None else "N/A"
        res_fmt = f"{twin_state.residual_kwh:+.2f} kWh" if twin_state.residual_kwh is not None else "N/A"
        dev_fmt = f"{twin_state.deviation_ratio * 100.0:+.1f}%" if twin_state.deviation_ratio is not None else "N/A"
        sev_fmt = twin_state.severity if twin_state.severity is not None else "N/A"
        typ_fmt = twin_state.anomaly_type if twin_state.anomaly_type is not None else "N/A"

        st.markdown(
            f"• **Actual Energy:** `{act_fmt}`\n\n"
            f"• **Expected Energy:** `{exp_fmt}`\n\n"
            f"• **Residual:** `{res_fmt}`\n\n"
            f"• **Relative Deviation:** `{dev_fmt}`\n\n"
            f"• **Severity:** `{sev_fmt}`\n\n"
            f"• **Anomaly Type:** `{typ_fmt}`"
        )

    with ts_col2:
        st.markdown("#### 🔮 Forecast State")
        hor_fmt = f"{twin_state.forecast_horizon_hours} hours" if twin_state.forecast_horizon_hours is not None else "N/A"
        tot_kwh_fmt = f"{twin_state.forecast_total_kwh:,.2f} kWh" if twin_state.forecast_total_kwh is not None else "N/A"
        peak_kwh_fmt = f"{twin_state.forecast_peak_kwh:.2f} kWh" if twin_state.forecast_peak_kwh is not None else "N/A"
        peak_ts_fmt = twin_state.forecast_peak_hour.strftime("%H:%M (%d %b)") if twin_state.forecast_peak_hour is not None else "N/A"
        w_src_fmt = twin_state.weather_source if twin_state.weather_source is not None else "N/A"

        st.markdown(
            f"• **Forecast Horizon:** `{hor_fmt}`\n\n"
            f"• **Next 24h Energy:** `{tot_kwh_fmt}`\n\n"
            f"• **Forecast Peak:** `{peak_kwh_fmt}`\n\n"
            f"• **Forecast Peak Hour:** `{peak_ts_fmt}`\n\n"
            f"• **Weather Source:** `{w_src_fmt}`"
        )

    with ts_col3:
        st.markdown("#### 💰 Impact State")
        cost_fmt = f"₹{twin_state.forecast_total_cost:,.2f}" if twin_state.forecast_total_cost is not None else "N/A"
        carbon_fmt = f"{twin_state.forecast_total_carbon_kg:,.2f} kg CO₂e" if twin_state.forecast_total_carbon_kg is not None else "N/A"

        st.markdown(
            f"• **Forecast Cost:** `{cost_fmt}`\n\n"
            f"• **Forecast Carbon:** `{carbon_fmt}`\n\n"
            f"• **Building ID:** `{twin_state.building_id}`\n\n"
            f"• **Meter ID:** `{twin_state.meter}`"
        )

    if twin_state.classification_reason:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.info(f"ℹ️ **Classification Reason:** {twin_state.classification_reason}")
else:
    st.warning("⚠️ **Building Twin State Unavailable:** Unable to aggregate building state.")

# ─────────────────────────────────────────
# PANEL 2E — WHAT-IF ENERGY SIMULATION
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>⚡ What-If Energy Simulation</div>", unsafe_allow_html=True)
st.caption(
    "Simulate controlled changes to the existing 24-hour energy forecast without retraining or rerunning the forecasting model."
)

sim_res = None
if forecast_24h_df is not None and len(forecast_24h_df) > 0:
    sim_col_sel, sim_col_val = st.columns([1.2, 1.0])

    with sim_col_sel:
        scenario_map = {
            "ENERGY_MULTIPLIER": "Energy Demand Multiplier (e.g. 1.10 = +10% demand)",
            "ENERGY_REDUCTION": "Energy Conservation / Reduction (e.g. 0.10 = 10% reduction)",
            "FIXED_KWH_ADDITION": "Fixed Base Load Addition (e.g. +5.0 kWh/hour)",
            "OCCUPANCY_CHANGE": "Occupancy Shift (e.g. +0.20 = +20% occupancy impact during business hours)",
            "TEMPERATURE_CHANGE": "Ambient Temperature Delta (e.g. +3.0°C cooling demand increase)",
            "EFFICIENCY_MULTIPLIER": "Equipment Efficiency Upgrade (e.g. 0.85 = 15% efficiency improvement)",
        }
        selected_scenario_key = st.selectbox(
            "Select Scenario Type",
            options=list(scenario_map.keys()),
            format_func=lambda x: scenario_map[x],
            index=0,
        )

    with sim_col_val:
        if selected_scenario_key == "ENERGY_MULTIPLIER":
            sim_value = st.number_input(
                "Multiplier Factor (e.g., 1.10 for +10%)",
                min_value=0.01,
                max_value=3.00,
                value=1.10,
                step=0.05,
                format="%.2f",
            )
            scenario_desc = f"+{(sim_value - 1.0) * 100.0:+.1f}% Energy Demand (Factor: {sim_value:.2f}x)" if sim_value >= 1.0 else f"{(sim_value - 1.0) * 100.0:.1f}% Energy Demand (Factor: {sim_value:.2f}x)"
        elif selected_scenario_key == "ENERGY_REDUCTION":
            sim_value = st.number_input(
                "Reduction Fraction (e.g., 0.10 for 10% reduction)",
                min_value=0.00,
                max_value=0.99,
                value=0.10,
                step=0.05,
                format="%.2f",
            )
            scenario_desc = f"{sim_value * 100.0:.1f}% Energy Reduction (Factor: {1.0 - sim_value:.2f}x)"
        elif selected_scenario_key == "OCCUPANCY_CHANGE":
            sim_value = st.number_input(
                "Occupancy Change Fraction (e.g., +0.20 for +20%, -0.20 for -20%)",
                min_value=-0.90,
                max_value=2.00,
                value=0.20,
                step=0.05,
                format="%.2f",
            )
            scenario_desc = f"{sim_value * 100.0:+.1f}% Occupancy Impact during Business Hours"
        elif selected_scenario_key == "TEMPERATURE_CHANGE":
            sim_value = st.number_input(
                "Temperature Delta in °C (e.g., +3.0°C, -2.0°C)",
                min_value=-15.0,
                max_value=15.0,
                value=3.0,
                step=0.5,
                format="%.1f",
            )
            scenario_desc = f"{sim_value:+.1f}°C Temperature Shift Scenario"
        elif selected_scenario_key == "EFFICIENCY_MULTIPLIER":
            sim_value = st.number_input(
                "Efficiency Multiplier (e.g., 0.85 for 15% efficiency gain)",
                min_value=0.10,
                max_value=1.50,
                value=0.85,
                step=0.05,
                format="%.2f",
            )
            scenario_desc = f"{(1.0 - sim_value) * 100.0:+.1f}% Efficiency Improvement (Multiplier: {sim_value:.2f}x)"
        else:  # FIXED_KWH_ADDITION
            sim_value = st.number_input(
                "Fixed kWh Addition per hour (e.g., +5.0 kWh)",
                min_value=-50.0,
                max_value=200.0,
                value=5.0,
                step=1.0,
                format="%.1f",
            )
            scenario_desc = f"{sim_value:+.1f} kWh/hour Fixed Addition"

    try:
        sim_res = sim_engine.simulate(
            forecast_df=forecast_24h_df,
            scenario_type=selected_scenario_key,
            value=sim_value,
        )

        st.info(
            f"🎯 **Active Scenario:** `{scenario_desc}`\n\n"
            f"ℹ️ *Simulation results are scenario estimates based on the existing 24-hour forecast.*"
        )

        # 1. Energy KPI Summary Cards
        sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4)

        with sim_m1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{sim_res['baseline_total_kwh']:,.1f}</div>
                <div class='metric-label'>Baseline 24h Energy</div>
                <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>kWh</div>
            </div>""", unsafe_allow_html=True)

        with sim_m2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#38bdf8'>{sim_res['simulated_total_kwh']:,.1f}</div>
                <div class='metric-label'>Simulated 24h Energy</div>
                <div style='color:#38bdf8;font-size:0.75rem;margin-top:4px'>kWh</div>
            </div>""", unsafe_allow_html=True)

        with sim_m3:
            delta_e_color = "#f97316" if sim_res["delta_total_kwh"] > 0 else "#86efac"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:{delta_e_color}'>{sim_res['delta_total_kwh']:+,.1f}</div>
                <div class='metric-label'>Change in Energy</div>
                <div style='color:{delta_e_color};font-size:0.75rem;margin-top:4px'>kWh ({sim_res['delta_total_percent']:+.1f}%)</div>
            </div>""", unsafe_allow_html=True)

        with sim_m4:
            peak_b_ts = sim_res["baseline_peak_hour"].strftime("%H:%M") if sim_res["baseline_peak_hour"] is not None else "N/A"
            peak_s_ts = sim_res["simulated_peak_hour"].strftime("%H:%M") if sim_res["simulated_peak_hour"] is not None else "N/A"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='font-size:1.3rem;padding-top:6px'>{sim_res['simulated_peak_kwh']:.1f} kWh</div>
                <div class='metric-label'>Simulated Peak (at {peak_s_ts})</div>
                <div style='color:#fbbf24;font-size:0.75rem;margin-top:4px'>Baseline Peak: {sim_res['baseline_peak_kwh']:.1f} kWh ({peak_b_ts})</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 2. Cost & Carbon Summary Cards
        cost_c1, cost_c2, cost_c3, carb_c1, carb_c2, carb_c3 = st.columns(6)

        with cost_c1:
            st.caption("Baseline Cost")
            st.markdown(f"**₹{sim_res['baseline_total_cost']:,.2f}**")

        with cost_c2:
            st.caption("Simulated Cost")
            st.markdown(f"**₹{sim_res['simulated_total_cost']:,.2f}**")

        with cost_c3:
            st.caption("Change in Cost")
            c_delta_color = "#f97316" if sim_res["delta_total_cost"] > 0 else "#86efac"
            st.markdown(f"<span style='color:{c_delta_color};font-weight:700;'>₹{sim_res['delta_total_cost']:+,.2f}</span>", unsafe_allow_html=True)

        with carb_c1:
            st.caption("Baseline Carbon")
            st.markdown(f"**{sim_res['baseline_total_carbon_kg']:,.2f} kg**")

        with carb_c2:
            st.caption("Simulated Carbon")
            st.markdown(f"**{sim_res['simulated_total_carbon_kg']:,.2f} kg**")

        with carb_c3:
            st.caption("Change in Carbon")
            cb_delta_color = "#f97316" if sim_res["delta_total_carbon_kg"] > 0 else "#86efac"
            st.markdown(f"<span style='color:{cb_delta_color};font-weight:700;'>{sim_res['delta_total_carbon_kg']:+,.2f} kg</span>", unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 3. Chart & Table Side-by-Side
        sim_chart_col, sim_table_col = st.columns([1.35, 1.0])
        hourly_sim_df = sim_res["hourly_df"]

        with sim_chart_col:
            st.caption("📈 **Hourly Forecast Comparison** (Baseline kWh vs Simulated kWh)")
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=hourly_sim_df["timestamp"],
                y=hourly_sim_df["baseline_kwh"],
                mode="lines",
                name="Baseline kWh",
                line=dict(color="#94a3b8", width=2, dash="dash"),
                hovertemplate="<b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Baseline:</b> %{y:.2f} kWh<extra></extra>",
            ))
            fig_sim.add_trace(go.Scatter(
                x=hourly_sim_df["timestamp"],
                y=hourly_sim_df["simulated_kwh"],
                mode="lines+markers",
                name="Simulated kWh",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=5),
                hovertemplate="<b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<br><b>Simulated:</b> %{y:.2f} kWh<extra></extra>",
            ))
            fig_sim.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                height=320,
                margin=dict(l=10, r=10, t=15, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Timestamp", gridcolor="#1e3a5f"),
                yaxis=dict(title="Energy Demand (kWh)", gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        with sim_table_col:
            st.caption("📋 **Hourly Simulation Trajectory**")
            display_sim_df = pd.DataFrame({
                "Timestamp": hourly_sim_df["timestamp"].dt.strftime("%d %b %H:%M"),
                "Baseline (kWh)": hourly_sim_df["baseline_kwh"].round(2),
                "Simulated (kWh)": hourly_sim_df["simulated_kwh"].round(2),
                "Delta (kWh)": hourly_sim_df["delta_kwh"].round(2),
                "Change (%)": hourly_sim_df["delta_percent"].round(1),
            })
            st.dataframe(display_sim_df, use_container_width=True, height=295, hide_index=True)

    except ValueError as val_err:
        st.error(f"⚠️ **Simulation Parameter Error:** {val_err}")
    except Exception as exc:
        st.error(f"⚠️ **Simulation Failed:** {exc}")

else:
    st.warning("⚠️ **What-If Simulation Unavailable:** A valid 24-hour energy forecast is required.")

# ─────────────────────────────────────────
# PANEL 2F — ENERGY RECOMMENDATIONS
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>💡 Energy Recommendations</div>", unsafe_allow_html=True)
st.caption(
    "Deterministic recommendations generated from the building's current Twin State, anomaly signals, "
    "forecast impact, and optional scenario results."
)

if twin_state is not None:
    try:
        rec_output = rec_engine.recommend(
            twin_state=twin_state,
            simulation_result=sim_res if (sim_res is not None) else None,
        )
        recommendations = rec_output.get("recommendations", [])
        gen_at = rec_output.get("generated_at", "N/A")

        st.caption(f"⏱️ **Recommendations Generated At (Software Timestamp):** `{gen_at}`")

        if len(recommendations) == 0:
            st.info("ℹ️ No recommendations available for the current state.")
        else:
            # Display recommendations in the exact priority order returned by engine
            for rec in recommendations:
                prio  = str(rec.get("priority", "INFO")).upper()
                cat   = rec.get("category", "ENERGY")
                title = rec.get("title", "")
                act   = rec.get("action", "")
                rsn   = rec.get("reason", "")
                evid  = rec.get("evidence", "")
                rid   = rec.get("recommendation_id", "")

                if prio == "CRITICAL":
                    css = "alert-high"
                    icon = "🔴"
                elif prio == "HIGH":
                    css = "alert-medium"
                    icon = "🟠"
                elif prio in ("MEDIUM", "LOW"):
                    css = "alert-low"
                    icon = "🟡" if prio == "MEDIUM" else "🔵"
                else:  # INFO
                    css = "alert-low"
                    icon = "🟢"

                st.markdown(
                    f"""<div class='{css}'>
                        <div>{icon} <b>[{rid}] {title}</b> &nbsp;|&nbsp; <b>Priority: {prio}</b> &nbsp;|&nbsp; <b>Category: {cat}</b></div>
                        <div style='font-size:0.9rem;margin-top:6px;'><b>Recommended Action:</b> {act}</div>
                        <div style='font-size:0.85rem;margin-top:4px;'><b>Reason:</b> {rsn}</div>
                        <div style='font-size:0.80rem;color:#cbd5e1;margin-top:4px;'><i>Evidence:</i> {evid}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
    except Exception as exc:
        st.error(f"⚠️ **Recommendation Generation Error:** {exc}")
else:
    st.warning("⚠️ **Energy Recommendations Unavailable:** Building Twin State is required.")

# ─────────────────────────────────────────
# PANEL 3 — ANOMALY ALERTS + HOURLY PATTERN
# ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='section-title'>🚨 Anomaly Alert Log</div>", unsafe_allow_html=True)
    st.caption("🎯 **PRIMARY SIGNAL:** Actual vs Expected Energy Consumption (Residual Engine)")

    if len(b_month) == 0:
        st.info(f"ℹ️ No meter observations recorded in {selected_month} 2016 for anomaly detection.")
    elif len(anomaly_events) == 0:
        st.markdown("<div class='alert-low'>✅ No residual anomalies detected this month</div>", unsafe_allow_html=True)
    else:
        # Display anomalies with residual metrics and severity
        for _, row in anomaly_events.head(8).iterrows():
            act     = row.get('actual_kwh', np.expm1(row.get('meter_reading_log', 0)))
            exp     = row.get('expected_kwh', act)
            res     = row.get('residual', act - exp)
            dev_pct = row.get('deviation_ratio', 0.0) * 100.0
            sev     = str(row.get('severity', 'NORMAL')).upper()
            ts      = pd.to_datetime(row['timestamp']).strftime('%d %b %H:%M')

            if sev in ('HIGH', 'CRITICAL'):
                css = 'alert-high'
                icon = '🔴'
            elif sev in ('MEDIUM', 'WARNING'):
                css = 'alert-medium'
                icon = '🟡'
            else:
                css = 'alert-low'
                icon = '🟢'

            direction_str = "above" if res >= 0 else "below"
            explanation = f"Actual consumption ({act:.1f} kWh) is {abs(dev_pct):.1f}% {direction_str} the expected consumption (Expected: {exp:.1f} kWh, Residual: {res:+.1f} kWh)."

            st.markdown(
                f"""<div class='{css}'>
                    <div>{icon} <b>{ts}</b> &nbsp;|&nbsp; <b>Severity: {sev}</b></div>
                    <div style='font-size:0.85rem;margin-top:4px;'>{explanation}</div>
                    <div style='font-size:0.75rem;color:#cbd5e1;margin-top:2px;'><i>Statistical residual divergence — does not prove physical causality.</i></div>
                </div>""",
                unsafe_allow_html=True
            )

        if len(anomaly_events) > 8:
            st.caption(f"+ {len(anomaly_events) - 8} more residual anomalies this month")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Explicit Auxiliary Isolation Forest Signal
    with st.expander("🛡️ Auxiliary Isolation Forest Signal (Secondary Evidence)"):
        st.caption("⚠️ **Training Scope Warning:**")
        st.warning(detector.TRAINING_SCOPE_NOTE)
        if len(b_month_anom) > 0 and 'isoforest_score' in b_month_anom.columns:
            iso_anoms = len(b_month_anom[b_month_anom['isoforest_anomaly'] == 1])
            avg_iso_score = b_month_anom['isoforest_score'].mean()
            st.markdown(
                f"• **Auxiliary Flags Triggered:** `{iso_anoms}` hours\n"
                f"• **Mean Score:** `{avg_iso_score:.4f}`\n\n"
                "*Note: Isolation Forest scores are provided strictly as auxiliary supporting evidence and are not used as the primary multi-building anomaly decision.*"
            )

with col_right:
    st.markdown("<div class='section-title'>🕐 Hourly Consumption Pattern</div>", unsafe_allow_html=True)

    if len(b_df) > 0:
        hourly_pattern = b_df.groupby('hour')['meter_reading'].median().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hourly_pattern['hour'], y=hourly_pattern['meter_reading'],
            mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(37,99,235,0.15)',
            line=dict(color='#3b82f6', width=2.5),
            marker=dict(size=5)
        ))
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.8)',
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title='Hour of Day', gridcolor='#1e3a5f', tickvals=list(range(0,24,3))),
            yaxis=dict(title='Median kWh', gridcolor='#1e3a5f')
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ No historical readings available for hourly pattern.")

# ─────────────────────────────────────────
# PANEL 4 — EXPLAINABILITY & LOCAL SHAP ATTRIBUTION
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🔍 Explainability & Local SHAP Attribution</div>", unsafe_allow_html=True)

col_exp_left, col_exp_right = st.columns([1, 1])

with col_exp_left:
    st.markdown("#### 🌲 Global Model Feature Importance (Tree Gain)")
    st.caption("Global tree gain measures overall reduction of training loss achieved by splits across all LightGBM trees. *Note: Global gain is not local SHAP attribution.*")

    try:
        feat_imp = explainer.get_tree_feature_importance(importance_type='gain').tail(12).sort_values('importance', ascending=True)

        fig3 = go.Figure(go.Bar(
            x=feat_imp['importance'],
            y=feat_imp['feature'],
            orientation='h',
            marker=dict(
                color=feat_imp['importance'],
                colorscale=[[0,'#1e3a5f'],[0.5,'#2563eb'],[1,'#60a5fa']]
            )
        ))
        fig3.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.8)',
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title='Feature Importance (Gain)', gridcolor='#1e3a5f'),
            yaxis=dict(gridcolor='#1e3a5f')
        )
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as exc:
        st.warning(f"⚠️ Could not compute feature importance: {exc}")

with col_exp_right:
    st.markdown("#### ⚡ Local SHAP Explanation for Energy Reading")
    st.caption("Decomposes a specific LightGBM prediction into additive feature contributions: $f(x) = E[f(x)] + \\sum \\phi_i$.")

    if len(b_month_clean) > 0:
        # Option to select an anomaly reading or peak reading
        if len(anomaly_events) > 0:
            anomaly_ts_list = anomaly_events['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
            selected_ts_str = st.selectbox("Select Anomaly Timestamp to Explain:", anomaly_ts_list, index=0)
            target_row = b_month_clean[b_month_clean['timestamp'].dt.strftime('%Y-%m-%d %H:%M') == selected_ts_str]
            if len(target_row) == 0:
                target_row = b_month_clean.iloc[[0]]
        else:
            st.caption("ℹ️ No anomalies this month. Explaining peak hourly reading:")
            target_row = b_month_clean.sort_values('actual', ascending=False).iloc[[0]]
            selected_ts_str = target_row['timestamp'].dt.strftime('%Y-%m-%d %H:%M').iloc[0]

        # Generate Local SHAP explanation for single selected row
        try:
            instance_exp = explainer.explain_instance(target_row.iloc[0])
            base_val = instance_exp["base_value"]
            pred_kwh = instance_exp["predicted_kwh"]
            act_kwh  = target_row["actual"].iloc[0] if "actual" in target_row.columns else np.expm1(target_row["meter_reading_log"].iloc[0])

            st.markdown(f"**Timestamp:** `{selected_ts_str}` &nbsp;|&nbsp; **Actual:** `{act_kwh:.1f} kWh` &nbsp;|&nbsp; **Model Expected:** `{pred_kwh:.1f} kWh`", unsafe_allow_html=True)
            st.caption(f"Base value $E[f(x)]$ = `{base_val:.3f}` log1p units (average model baseline)")

            pos_drivers = instance_exp["top_positive_drivers"]
            neg_drivers = instance_exp["top_negative_drivers"]

            st.markdown("##### 🔺 Positive Contributors (Pushed Prediction Higher)")
            if pos_drivers:
                for d in pos_drivers[:4]:
                    val_fmt = f"{d['value']:.2f}" if isinstance(d['value'], float) else str(d['value'])
                    st.markdown(f"• **`{d['feature']}`** (`{val_fmt}`): `+{d['shap_value']:.3f}` SHAP value (contributed to higher model prediction)")
            else:
                st.caption("No significant positive drivers.")

            st.markdown("##### 🔻 Negative Contributors (Pushed Prediction Lower)")
            if neg_drivers:
                for d in neg_drivers[:4]:
                    val_fmt = f"{d['value']:.2f}" if isinstance(d['value'], float) else str(d['value'])
                    st.markdown(f"• **`{d['feature']}`** (`{val_fmt}`): `{d['shap_value']:.3f}` SHAP value (contributed to lower model prediction)")
            else:
                st.caption("No significant negative drivers.")

            st.caption("ℹ️ *Interpretation Note: SHAP value reflects local model attribution for the LightGBM prediction. It does NOT claim physical causality in the real building system.*")
        except Exception as exc:
            st.warning(f"⚠️ Could not compute local SHAP explanation: {exc}")
    else:
        st.info(f"ℹ️ Local SHAP instance explanation requires observed meter records for {selected_month} 2016.")

# ─────────────────────────────────────────
# PANEL 5 — MONTHLY SUMMARY
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>📅 Monthly Consumption Overview — Full Year</div>", unsafe_allow_html=True)

@st.cache_data
def load_monthly_summary():
    return loader.get_monthly_summary(meter=0)

monthly_summary = load_monthly_summary().copy()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_summary['month_name'] = monthly_summary['month'].apply(lambda x: month_labels[int(x)-1])

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=monthly_summary['month_name'], y=monthly_summary['total'],
    name='Total kWh', marker_color='#2563eb', opacity=0.85
))
fig4.add_trace(go.Scatter(
    x=monthly_summary['month_name'], y=monthly_summary['peak'],
    mode='lines+markers', name='Peak kWh',
    line=dict(color='#f97316', width=2),
    marker=dict(size=6), yaxis='y2'
))

# Highlight selected month
selected_idx = month_num - 1
fig4.add_vline(
    x=selected_idx, line_dash='dash',
    line_color='#fbbf24', opacity=0.6,
    annotation_text=f'{selected_month}',
    annotation_position='top'
)

fig4.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.8)',
    height=300,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(gridcolor='#1e3a5f'),
    yaxis=dict(title='Total kWh', gridcolor='#1e3a5f'),
    yaxis2=dict(title='Peak kWh', overlaying='y', side='right', gridcolor='#1e3a5f')
)
st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:0.8rem'>Twinergy — AI-Driven Digital Twin for Building Energy Management &nbsp;|&nbsp; "
    "ASHRAE GEPIII Dataset &nbsp;|&nbsp; LightGBM + Isolation Forest + SHAP &nbsp;|&nbsp; "
    "IoT–Cloud Integration Project 2024–25</center>",
    unsafe_allow_html=True
)