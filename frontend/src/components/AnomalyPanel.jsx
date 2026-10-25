import React, { useState } from 'react'
import { AlertOctagon, Info, ShieldAlert, Filter, AlertTriangle } from 'lucide-react'
import EmptyState from './common/EmptyState'

export default function AnomalyPanel({ anomalyData }) {
  const [showAnomaliesOnly, setShowAnomaliesOnly] = useState(false)

  if (!anomalyData) {
    return (
      <div className="glass-card">
        <div className="section-header">
          <div className="section-title">
            <AlertOctagon size={20} /> Energy Anomalies Analysis
          </div>
        </div>
        <EmptyState title="No Anomaly Data" message="Anomaly records are currently unavailable for this building." />
      </div>
    )
  }

  const {
    window_start,
    window_end,
    total_records,
    anomaly_count,
    primary_residual_signal,
    auxiliary_isolation_forest_signal,
    anomalies = [],
  } = anomalyData

  const displayedAnomalies = showAnomaliesOnly
    ? anomalies.filter((a) => a.is_anomaly === 1)
    : anomalies

  // Format helper
  const formatDate = (isoStr) => {
    if (!isoStr) return ''
    try {
      const d = new Date(isoStr)
      return `${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })} · ${d.toLocaleDateString([], { day: 'numeric', month: 'short' })}`
    } catch {
      return isoStr
    }
  }

  const getSeverityBadge = (sev) => {
    switch (sev) {
      case 'NORMAL':
        return <span className="badge badge-normal">Normal</span>
      case 'WARNING':
        return <span className="badge badge-warning">Warning</span>
      case 'CRITICAL':
        return <span className="badge badge-critical">Critical</span>
      default:
        return <span className="badge badge-unavailable">{sev || 'N/A'}</span>
    }
  }

  const getTypeBadge = (type) => {
    switch (type) {
      case 'RAPID_SPIKE':
        return <span className="badge badge-critical">⚡ Rapid Spike</span>
      case 'PERSISTENT_HIGH':
        return <span className="badge badge-warning">📈 Persistent High</span>
      case 'HIGH_DEMAND':
        return <span className="badge badge-warning">▲ High Demand</span>
      case 'LOW_DEMAND':
        return <span className="badge badge-normal">▼ Low Demand</span>
      case 'NORMAL':
        return <span className="badge badge-normal">Normal</span>
      default:
        return <span className="badge badge-unavailable">{type || 'N/A'}</span>
    }
  }

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <AlertOctagon size={20} /> Energy Anomalies & Explainable Classification
          </div>
          <div className="section-caption">
            Primary residual engine compares observed consumption against expected baselines. Isolation Forest provided as secondary signal.
          </div>
        </div>

        {/* Filter Toggle */}
        <button
          onClick={() => setShowAnomaliesOnly(!showAnomaliesOnly)}
          className={`btn ${showAnomaliesOnly ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '6px 14px', fontSize: '0.8rem' }}
          aria-label="Toggle Flagged Anomalies Only"
        >
          <Filter size={14} /> {showAnomaliesOnly ? 'Show All Hours' : 'Flagged Only'}
        </button>
      </div>

      {/* Signal Architecture Comparison Grid */}
      <div className="grid-2col" style={{ marginBottom: 20 }}>
        {/* Signal 1: Primary Residual Engine */}
        <div
          style={{
            background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85))',
            borderRadius: '10px',
            padding: '16px 18px',
            border: '1px solid rgba(56, 189, 248, 0.3)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ fontWeight: 700, color: 'var(--cyan-primary)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldAlert size={16} /> PRIMARY SIGNAL: Residual Engine
            </div>
            <span className="badge badge-info">{anomaly_count} Anomalies Detected</span>
          </div>

          <p style={{ fontSize: '0.78rem', color: '#cbd5e1', marginBottom: 12 }}>
            Evaluates relative consumption deviations with deterministic severity thresholds (Warning &gt; 25%, Critical &gt; 50%).
          </p>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
            {primary_residual_signal?.counts_by_severity &&
              Object.entries(primary_residual_signal.counts_by_severity).map(([sev, count]) => (
                <div
                  key={sev}
                  style={{
                    fontSize: '0.75rem',
                    background: 'rgba(15, 23, 42, 0.6)',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <span style={{ color: '#94a3b8' }}>{sev}:</span> <strong>{count}</strong>
                </div>
              ))}
          </div>

          <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontStyle: 'italic', display: 'flex', alignItems: 'flex-start', gap: 6 }}>
            <Info size={14} style={{ flexShrink: 0, marginTop: 1, color: '#38bdf8' }} />
            <span>{primary_residual_signal?.causality_warning}</span>
          </div>
        </div>

        {/* Signal 2: Auxiliary Isolation Forest */}
        <div
          style={{
            background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.75))',
            borderRadius: '10px',
            padding: '16px 18px',
            border: '1px solid rgba(251, 191, 36, 0.25)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ fontWeight: 700, color: 'var(--amber-primary)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: 6 }}>
              <AlertTriangle size={16} /> AUXILIARY SIGNAL: Isolation Forest
            </div>
            <span className="badge badge-warning">
              {auxiliary_isolation_forest_signal?.anomaly_count || 0} Flags
            </span>
          </div>

          <div style={{ fontSize: '0.8rem', color: '#cbd5e1', marginBottom: 8 }}>
            <strong>Scope Limitation:</strong> {auxiliary_isolation_forest_signal?.training_scope}
          </div>

          <div style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.4, background: 'rgba(15, 23, 42, 0.6)', padding: '8px 12px', borderRadius: '6px' }}>
            {auxiliary_isolation_forest_signal?.warning}
          </div>
        </div>
      </div>

      {/* Anomalies Table */}
      {displayedAnomalies.length === 0 ? (
        <EmptyState
          title={showAnomaliesOnly ? 'No Anomaly Events Found' : 'No Observations'}
          message={
            showAnomaliesOnly
              ? 'All hourly readings in this window are operating within normal baseline ranges.'
              : 'No hourly observations available for this window.'
          }
        />
      ) : (
        <div className="data-table-container" style={{ maxHeight: 380 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Actual Demand</th>
                <th>Expected Demand</th>
                <th>Residual (Δ)</th>
                <th>Deviation</th>
                <th>Severity</th>
                <th>Classification</th>
                <th>Reason</th>
                <th>Aux Signal</th>
              </tr>
            </thead>
            <tbody>
              {displayedAnomalies.map((a, idx) => (
                <tr
                  key={`anom-${idx}`}
                  style={{
                    backgroundColor: a.is_anomaly === 1 ? 'rgba(244, 63, 94, 0.05)' : 'transparent',
                  }}
                >
                  <td style={{ whiteSpace: 'nowrap' }}>{formatDate(a.timestamp)}</td>
                  <td style={{ fontWeight: 600, color: '#f8fafc' }}>{a.actual_kwh.toFixed(1)} kWh</td>
                  <td style={{ color: '#94a3b8' }}>{a.expected_kwh.toFixed(1)} kWh</td>
                  <td
                    style={{
                      fontWeight: 600,
                      color: a.residual > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)',
                    }}
                  >
                    {a.residual > 0 ? '+' : ''}
                    {a.residual.toFixed(1)} kWh
                  </td>
                  <td>
                    <span style={{ fontWeight: 600 }}>{(a.deviation_ratio * 100).toFixed(1)}%</span>
                  </td>
                  <td>{getSeverityBadge(a.severity)}</td>
                  <td>{getTypeBadge(a.anomaly_type)}</td>
                  <td style={{ fontSize: '0.78rem', color: '#cbd5e1', maxWidth: 280 }}>
                    {a.classification_reason}
                  </td>
                  <td>
                    {a.isoforest_anomaly === 1 ? (
                      <span className="badge badge-warning" style={{ fontSize: '0.7rem' }}>
                        IF Flagged
                      </span>
                    ) : (
                      <span style={{ color: '#64748b', fontSize: '0.75rem' }}>Nominal</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
