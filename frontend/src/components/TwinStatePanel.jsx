import React from 'react'
import { Cpu, Clock, Zap, Activity, TrendingUp, IndianRupee, Sun, ShieldCheck } from 'lucide-react'
import EmptyState from './common/EmptyState'

export default function TwinStatePanel({ stateData }) {
  if (!stateData) {
    return (
      <div className="glass-card">
        <div className="section-header">
          <div className="section-title">
            <Cpu size={20} /> Building Digital Twin State
          </div>
        </div>
        <EmptyState title="Twin State Unavailable" message="Software digital twin representation could not be aggregated for this building." />
      </div>
    )
  }

  const {
    building_id,
    meter,
    timestamp,
    state_status,
    actual_kwh,
    expected_kwh,
    residual_kwh,
    deviation_ratio,
    severity,
    anomaly_type,
    classification_reason,
    forecast_horizon_hours,
    forecast_total_kwh,
    forecast_peak_kwh,
    forecast_peak_hour,
    forecast_total_cost,
    forecast_total_carbon_kg,
    weather_source,
    generated_at,
  } = stateData

  const formatDate = (isoStr) => {
    if (!isoStr) return 'N/A'
    try {
      const d = new Date(isoStr)
      return `${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })} · ${d.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' })}`
    } catch {
      return isoStr
    }
  }

  const formatKwh = (val) => (typeof val === 'number' && !isNaN(val) ? `${val.toFixed(2)} kWh` : 'N/A')
  const formatCost = (val) => (typeof val === 'number' && !isNaN(val) ? `₹${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A')
  const formatCarbon = (val) => (typeof val === 'number' && !isNaN(val) ? `${val.toFixed(2)} kg CO₂e` : 'N/A')
  const formatPercent = (val) => (typeof val === 'number' && !isNaN(val) ? `${(val * 100).toFixed(1)}%` : 'N/A')

  // Status banner styling
  const getStatusBannerStyle = (status) => {
    switch (status) {
      case 'NORMAL':
        return { bg: 'rgba(5, 46, 22, 0.7)', border: '#16a34a', color: '#86efac', text: 'NORMAL OPERATION' }
      case 'WARNING':
        return { bg: 'rgba(67, 20, 7, 0.7)', border: '#f97316', color: '#fdba74', text: 'ELEVATED CONSUMPTION (WARNING)' }
      case 'CRITICAL':
        return { bg: 'rgba(69, 10, 10, 0.7)', border: '#dc2626', color: '#fca5a5', text: 'CRITICAL CONSUMPTION ANOMALY' }
      default:
        return { bg: 'rgba(30, 41, 59, 0.7)', border: '#64748b', color: '#cbd5e1', text: 'DATA UNAVAILABLE' }
    }
  }

  const banner = getStatusBannerStyle(state_status)

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <Cpu size={20} /> Building Digital Twin State
          </div>
          <div className="section-caption">
            Aggregated software state model integrating observations, recursive forecast, residual anomaly signals, and cost/carbon impacts.
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: '#94a3b8' }}>
          <Clock size={14} /> Ref: {formatDate(timestamp)}
        </div>
      </div>

      {/* State Status Banner */}
      <div
        style={{
          background: banner.bg,
          borderLeft: `4px solid ${banner.border}`,
          borderRadius: '8px',
          padding: '14px 18px',
          marginBottom: 20,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <div>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: banner.color, display: 'flex', alignItems: 'center', gap: 8 }}>
            <ShieldCheck size={18} /> {banner.text}
          </div>
          <div style={{ fontSize: '0.78rem', color: '#e2e8f0', marginTop: 4 }}>
            Building ID: <strong>#{building_id}</strong> · Meter: <strong>#{meter}</strong> · State Generated: <strong>{formatDate(generated_at)}</strong>
          </div>
        </div>
        <div style={{ fontSize: '0.75rem', background: 'rgba(0, 0, 0, 0.3)', padding: '4px 10px', borderRadius: '6px', color: '#cbd5e1' }}>
          Software Twin (Non-telemetric)
        </div>
      </div>

      {/* 3-Column State Breakdown */}
      <div className="grid-2col" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {/* Column 1: Current Energy & Anomaly */}
        <div style={{ background: 'rgba(15, 23, 42, 0.65)', padding: '18px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--cyan-primary)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Zap size={16} /> 1. Energy & Anomaly State
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Actual Reading:</span>
              <strong style={{ color: '#f8fafc' }}>{formatKwh(actual_kwh)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Expected Baseline:</span>
              <strong style={{ color: '#f8fafc' }}>{formatKwh(expected_kwh)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Residual (Δ):</span>
              <strong style={{ color: residual_kwh > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)' }}>
                {residual_kwh !== null && residual_kwh !== undefined ? `${residual_kwh > 0 ? '+' : ''}${residual_kwh.toFixed(2)} kWh` : 'N/A'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Deviation Ratio:</span>
              <strong>{formatPercent(deviation_ratio)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Anomaly Severity:</span>
              <strong>{severity || 'N/A'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Classification Type:</span>
              <strong>{anomaly_type || 'N/A'}</strong>
            </div>
          </div>
        </div>

        {/* Column 2: Forecast Horizon State */}
        <div style={{ background: 'rgba(15, 23, 42, 0.65)', padding: '18px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#38bdf8', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <TrendingUp size={16} /> 2. Forecast State (24h)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Horizon Duration:</span>
              <strong>{forecast_horizon_hours ? `${forecast_horizon_hours} Hours` : 'N/A'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Cumulative Energy:</span>
              <strong style={{ color: '#38bdf8' }}>{formatKwh(forecast_total_kwh)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Peak Demand:</span>
              <strong style={{ color: 'var(--amber-primary)' }}>{formatKwh(forecast_peak_kwh)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Peak Demand Hour:</span>
              <strong>{formatDate(forecast_peak_hour)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Weather Source:</span>
              <strong>{weather_source || 'N/A'}</strong>
            </div>
          </div>
        </div>

        {/* Column 3: Impact Projections State */}
        <div style={{ background: 'rgba(15, 23, 42, 0.65)', padding: '18px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--amber-primary)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <IndianRupee size={16} /> 3. Impact Projections (24h)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Projected Cost:</span>
              <strong style={{ color: 'var(--amber-primary)' }}>{formatCost(forecast_total_cost)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Projected Carbon:</span>
              <strong style={{ color: 'var(--emerald-primary)' }}>{formatCarbon(forecast_total_carbon_kg)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Target Meter:</span>
              <strong>Meter #{meter}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Building ID:</span>
              <strong>Building #{building_id}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Classification Reason Callout */}
      {classification_reason && (
        <div style={{ marginTop: 16, background: 'rgba(15, 23, 42, 0.7)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '10px 14px', fontSize: '0.82rem', color: '#cbd5e1' }}>
          <strong>State Classification Reason:</strong> {classification_reason}
        </div>
      )}
    </div>
  )
}
