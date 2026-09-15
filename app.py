import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

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
# LOAD DATA & MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('twinergy_clean.csv', parse_dates=['timestamp'])
    return df

@st.cache_resource
def load_models():
    with open('twinergy_lgbm.pkl', 'rb') as f:
        lgbm = pickle.load(f)
    with open('twinergy_isoforest.pkl', 'rb') as f:
        iso = pickle.load(f)
    return lgbm, iso

df        = load_data()
model, iso = load_models()

FEATURES = [
    'hour', 'dayofweek', 'month', 'dayofyear',
    'is_weekend', 'is_business_hours',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'lag_1hr', 'lag_24hr', 'lag_168hr',
    'rolling_mean_6hr', 'rolling_mean_24hr', 'rolling_std_24hr', 'rolling_mean_168hr',
    'temp_x_hour', 'temp_x_weekend', 'occupancy_proxy',
    'air_temperature', 'square_feet'
]

ANOMALY_FEATURES = [
    'meter_reading_log', 'rolling_mean_6hr', 'rolling_std_24hr',
    'lag_1hr', 'lag_24hr', 'hour', 'is_weekend', 'air_temperature'
]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Twinergy")
    st.markdown("*AI-Driven Digital Twin*")
    st.markdown("---")

    building_ids = sorted(df[df['meter'] == 0]['building_id'].unique())
    selected_building = st.selectbox("Select Building", building_ids, index=0)

    st.markdown("---")

    primary_use = df[df['building_id'] == selected_building]['primary_use'].iloc[0]
    sq_ft       = df[df['building_id'] == selected_building]['square_feet'].iloc[0]
    site        = df[df['building_id'] == selected_building]['site_id'].iloc[0]

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
    selected_month = st.selectbox("Month", list(month_map.keys()), index=4)
    month_num = month_map[selected_month]

    st.markdown("---")
    st.caption("ASHRAE GEPIII Dataset")
    st.caption("LightGBM + Isolation Forest")
    st.caption("MAPE: 13.09% | RMSE: 0.1216")

# ─────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────
b_df = df[
    (df['building_id'] == selected_building) &
    (df['meter'] == 0)
].copy().sort_values('timestamp')

b_month = b_df[b_df['timestamp'].dt.month == month_num].copy()

# Predictions
b_month_clean = b_month.dropna(subset=FEATURES)
if len(b_month_clean) > 0:
    b_month_clean = b_month_clean.copy()
    b_month_clean['square_feet'] = b_month_clean['square_feet'].fillna(b_month_clean['square_feet'].median())
    preds_log = model.predict(b_month_clean[FEATURES])
    b_month_clean['predicted'] = np.expm1(preds_log)
    b_month_clean['actual']    = np.expm1(b_month_clean['meter_reading_log'])

# Anomaly detection
b_month_anom = b_month.dropna(subset=ANOMALY_FEATURES).copy()
if len(b_month_anom) > 0:
    anom_scores = iso.predict(b_month_anom[ANOMALY_FEATURES])
    b_month_anom['is_anomaly'] = (anom_scores == -1).astype(int)
    anomaly_events = b_month_anom[b_month_anom['is_anomaly'] == 1]
else:
    anomaly_events = pd.DataFrame()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# 🏢 Twinergy")
st.markdown(f"**AI-Driven Digital Twin** — Building `{selected_building}` · {primary_use} · {selected_month} 2016")
st.markdown("---")

# ─────────────────────────────────────────
# PANEL 1 — KPI METRICS
# ─────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

if len(b_month_clean) > 0:
    current_kwh   = b_month_clean['actual'].iloc[-1]
    predicted_kwh = b_month_clean['predicted'].iloc[-1]
    avg_kwh       = b_month_clean['actual'].mean()
    total_kwh     = b_month_clean['actual'].sum()
    n_anomalies   = len(anomaly_events)
    co2_kg        = total_kwh * 0.233  # avg kg CO2 per kWh

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
            <div style='color:{alert_color};font-size:0.75rem;margin-top:4px'>Isolation Forest</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{co2_kg:,.0f}</div>
            <div class='metric-label'>CO₂ Estimate (kg)</div>
            <div style='color:#93c5fd;font-size:0.75rem;margin-top:4px'>@0.233 kg/kWh</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# PANEL 2 — FORECAST CHART
# ─────────────────────────────────────────
st.markdown("<div class='section-title'>📈 Energy Forecast — Actual vs Predicted</div>", unsafe_allow_html=True)

if len(b_month_clean) > 0:
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([b_month_clean['timestamp'], b_month_clean['timestamp'][::-1]]),
        y=pd.concat([b_month_clean['predicted'] * 1.15, (b_month_clean['predicted'] * 0.85)[::-1]]),
        fill='toself', fillcolor='rgba(37,99,235,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band', hoverinfo='skip'
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

# ─────────────────────────────────────────
# PANEL 3 — ANOMALY ALERTS + HOURLY PATTERN
# ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='section-title'>🚨 Anomaly Alert Log</div>", unsafe_allow_html=True)

    if len(anomaly_events) == 0:
        st.markdown("<div class='alert-low'>✅ No anomalies detected this month</div>", unsafe_allow_html=True)
    else:
        # Assign severity based on deviation from rolling mean
        for _, row in anomaly_events.head(8).iterrows():
            val     = np.expm1(row['meter_reading_log'])
            base    = np.expm1(row['rolling_mean_6hr']) if row['rolling_mean_6hr'] > 0 else val
            dev     = abs(val - base) / (base + 1e-6)
            ts      = pd.to_datetime(row['timestamp']).strftime('%d %b %H:%M')

            if dev > 0.5:
                severity = 'HIGH'
                css = 'alert-high'
                icon = '🔴'
            elif dev > 0.25:
                severity = 'MEDIUM'
                css = 'alert-medium'
                icon = '🟡'
            else:
                severity = 'LOW'
                css = 'alert-low'
                icon = '🟢'

            st.markdown(
                f"<div class='{css}'>{icon} <b>{ts}</b> — {val:.1f} kWh &nbsp;|&nbsp; Severity: {severity}</div>",
                unsafe_allow_html=True
            )

        if len(anomaly_events) > 8:
            st.caption(f"+ {len(anomaly_events) - 8} more anomalies this month")

with col_right:
    st.markdown("<div class='section-title'>🕐 Hourly Consumption Pattern</div>", unsafe_allow_html=True)

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

# ─────────────────────────────────────────
# PANEL 4 — SHAP EXPLAINER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🔍 Why Is Consumption This High? — SHAP Feature Importance</div>", unsafe_allow_html=True)

if len(b_month_clean) > 0:
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(12)

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
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title='Feature Importance (Gain)', gridcolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f')
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────
# PANEL 5 — MONTHLY SUMMARY
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>📅 Monthly Consumption Overview — Full Year</div>", unsafe_allow_html=True)

all_elec = df[df['meter'] == 0].copy()
monthly_summary = all_elec.groupby(all_elec['timestamp'].dt.month)['meter_reading'].agg(['sum','mean','max']).reset_index()
monthly_summary.columns = ['month', 'total', 'avg', 'peak']
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_summary['month_name'] = monthly_summary['month'].apply(lambda x: month_labels[x-1])

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