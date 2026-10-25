import React, { useState } from 'react'
import { TrendingUp, Table, LineChart as ChartIcon, Sun, Calendar } from 'lucide-react'
import LineChart from './charts/LineChart'
import EmptyState from './common/EmptyState'

export default function ForecastPanel({ forecastData }) {
  const [viewMode, setViewMode] = useState('chart') // 'chart' | 'table'

  if (!forecastData || !forecastData.hourly_forecast || forecastData.hourly_forecast.length === 0) {
    return (
      <div className="glass-card">
        <div className="section-header">
          <div className="section-title">
            <TrendingUp size={20} /> 24-Hour Energy Forecast
          </div>
        </div>
        <EmptyState title="Forecast Unavailable" message="No recursive 24-hour forecast data is available for this building." />
      </div>
    )
  }

  const { summary, weather_source, cutoff_time, hourly_forecast } = forecastData

  // Time formatters
  const formatHour = (isoStr) => {
    if (!isoStr) return ''
    try {
      const d = new Date(isoStr)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
    } catch {
      return isoStr.substring(11, 16) || isoStr
    }
  }

  const formatDateHour = (isoStr) => {
    if (!isoStr) return 'N/A'
    try {
      const d = new Date(isoStr)
      return `${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })} (${d.toLocaleDateString([], { day: 'numeric', month: 'short' })})`
    } catch {
      return isoStr
    }
  }

  const chartSeries = [
    {
      key: 'predicted_kwh',
      name: 'Predicted Demand',
      color: '#38bdf8',
      strokeWidth: 2.5,
      areaFill: true,
    },
  ]

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <TrendingUp size={20} /> 24-Hour Recursive Energy Forecast
          </div>
          <div className="section-caption">
            Recursive multi-step forecast starting strictly after cutoff ({formatDateHour(cutoff_time)}). Autoregressively feeds prior predictions without target leakage.
          </div>
        </div>

        {/* View Toggle */}
        <div style={{ display: 'flex', gap: 6, background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '8px' }}>
          <button
            onClick={() => setViewMode('chart')}
            className={`btn ${viewMode === 'chart' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            aria-label="View Forecast Chart"
          >
            <ChartIcon size={14} /> Chart
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`btn ${viewMode === 'table' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            aria-label="View Forecast Table"
          >
            <Table size={14} /> Table
          </button>
        </div>
      </div>

      {/* Summary KPI Badges */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 12,
          marginBottom: 20,
        }}
      >
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>24h Total Demand</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8', marginTop: 2 }}>
            {summary.total_kwh.toLocaleString(undefined, { maximumFractionDigits: 1 })} kWh
          </div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Peak Hourly Demand</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--amber-primary)', marginTop: 2 }}>
            {summary.peak_kwh.toFixed(1)} kWh
          </div>
          <div style={{ fontSize: '0.75rem', color: '#cbd5e1', marginTop: 2 }}>
            at {formatDateHour(summary.peak_timestamp)}
          </div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Minimum Demand</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--emerald-primary)', marginTop: 2 }}>
            {summary.min_kwh.toFixed(1)} kWh
          </div>
          <div style={{ fontSize: '0.75rem', color: '#cbd5e1', marginTop: 2 }}>
            at {formatDateHour(summary.min_timestamp)}
          </div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Weather Source</div>
          <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sun size={15} color="var(--amber-primary)" />
            {weather_source}
          </div>
        </div>
      </div>

      {/* Main View: Chart or Table */}
      {viewMode === 'chart' ? (
        <div>
          <LineChart
            data={hourly_forecast}
            series={chartSeries}
            xKey="timestamp"
            xFormatter={formatHour}
            yFormatter={(v) => v.toFixed(1)}
            yUnit="kWh"
            height={280}
            highlightPeak={true}
            peakKey="predicted_kwh"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12, fontSize: '0.78rem', color: '#64748b' }}>
            <span>🟡 Peak demand indicator</span>
            <span>Hover over any hour to inspect exact prediction and temperature</span>
          </div>
        </div>
      ) : (
        <div className="data-table-container" style={{ maxHeight: 320 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Timestamp</th>
                <th>Predicted Demand (kWh)</th>
                <th>Log1p Value</th>
                <th>Air Temp (°C)</th>
              </tr>
            </thead>
            <tbody>
              {hourly_forecast.map((h) => (
                <tr key={`fc-step-${h.step}`}>
                  <td>#{h.step}</td>
                  <td>{formatDateHour(h.timestamp)}</td>
                  <td style={{ fontWeight: 600, color: '#38bdf8' }}>{h.predicted_kwh.toFixed(2)}</td>
                  <td>{h.predicted_log.toFixed(3)}</td>
                  <td>{h.air_temperature !== null && h.air_temperature !== undefined ? `${h.air_temperature.toFixed(1)} °C` : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
