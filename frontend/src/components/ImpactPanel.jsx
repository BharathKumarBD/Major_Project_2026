import React, { useState } from 'react'
import { IndianRupee, Cloud, AlertCircle, TrendingUp, Sliders, RefreshCw } from 'lucide-react'
import LineChart from './charts/LineChart'
import EmptyState from './common/EmptyState'

export default function ImpactPanel({ impactData, onUpdateAssumptions, isUpdating = false }) {
  const [customTariff, setCustomTariff] = useState('')
  const [customCarbon, setCustomCarbon] = useState('')

  if (!impactData) {
    return (
      <div className="glass-card">
        <div className="section-header">
          <div className="section-title">
            <IndianRupee size={20} /> Cost & Carbon Impact
          </div>
        </div>
        <EmptyState title="Impact Data Unavailable" message="24-hour cost and carbon calculations require an active forecast." />
      </div>
    )
  }

  const { assumptions, summary, hourly_impact = [] } = impactData

  // Handlers for updating assumptions
  const handleApplyAssumptions = (e) => {
    e.preventDefault()
    const tVal = customTariff ? parseFloat(customTariff) : null
    const cVal = customCarbon ? parseFloat(customCarbon) : null
    if (onUpdateAssumptions) {
      onUpdateAssumptions(tVal, cVal)
    }
  }

  const handleResetAssumptions = () => {
    setCustomTariff('')
    setCustomCarbon('')
    if (onUpdateAssumptions) {
      onUpdateAssumptions(null, null)
    }
  }

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

  // Cost trajectory series
  const costSeries = [
    {
      key: 'hourly_cost',
      name: 'Hourly Cost',
      color: '#fbbf24',
      strokeWidth: 2.2,
      areaFill: true,
    },
  ]

  // Carbon trajectory series
  const carbonSeries = [
    {
      key: 'hourly_carbon_kg',
      name: 'Hourly Carbon',
      color: '#34d399',
      strokeWidth: 2.2,
      areaFill: true,
    },
  ]

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <IndianRupee size={20} /> 24-Hour Cost & Carbon Impact
          </div>
          <div className="section-caption">
            Financial cost and carbon emissions derived from the validated 24-hour energy forecast under configurable tariff parameters.
          </div>
        </div>
      </div>

      {/* Assumptions Notice */}
      <div className="alert-banner alert-info" style={{ marginBottom: 20 }}>
        <AlertCircle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div style={{ flex: 1, fontSize: '0.82rem' }}>
          <strong>Estimates & Assumptions:</strong> Cost calculated at ₹{assumptions?.tariff_per_kwh?.toFixed(2)}/kWh. Carbon intensity estimated at {assumptions?.carbon_intensity_kg_per_kwh?.toFixed(2)} kg CO₂e/kWh. {assumptions?.disclaimer}
        </div>
      </div>

      {/* KPI Cards Row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 14,
          marginBottom: 24,
        }}
      >
        <div className="kpi-card">
          <div className="kpi-label">
            <IndianRupee size={15} color="var(--amber-primary)" /> Total 24h Cost
          </div>
          <div className="kpi-value" style={{ color: 'var(--amber-primary)' }}>
            ₹{summary.total_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className="kpi-subtext">Peak Cost: {formatDateHour(summary.peak_cost_hour)}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <Cloud size={15} color="var(--emerald-primary)" /> Total 24h Carbon
          </div>
          <div className="kpi-value" style={{ color: 'var(--emerald-primary)' }}>
            {summary.total_carbon_kg.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg
          </div>
          <div className="kpi-subtext">Peak Carbon: {formatDateHour(summary.peak_carbon_hour)}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Average Hourly Cost</div>
          <div className="kpi-value" style={{ fontSize: '1.45rem', paddingTop: 2 }}>
            ₹{summary.average_hourly_cost.toFixed(2)}
          </div>
          <div className="kpi-subtext">Mean per-hour expenditure</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Average Hourly Carbon</div>
          <div className="kpi-value" style={{ fontSize: '1.45rem', paddingTop: 2 }}>
            {summary.average_hourly_carbon_kg.toFixed(2)} kg
          </div>
          <div className="kpi-subtext">Mean per-hour emissions</div>
        </div>
      </div>

      {/* Trajectory Charts Side-by-Side */}
      <div className="grid-2col" style={{ marginBottom: 20 }}>
        {/* Cost Chart */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fbbf24', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <IndianRupee size={15} /> Hourly Energy Cost Trajectory (₹)
          </div>
          <LineChart
            data={hourly_impact}
            series={costSeries}
            xKey="timestamp"
            xFormatter={formatHour}
            yFormatter={(v) => `₹${v.toFixed(0)}`}
            yUnit=""
            height={200}
          />
        </div>

        {/* Carbon Chart */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34d399', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cloud size={15} /> Hourly Carbon Emissions Trajectory (kg CO₂e)
          </div>
          <LineChart
            data={hourly_impact}
            series={carbonSeries}
            xKey="timestamp"
            xFormatter={formatHour}
            yFormatter={(v) => v.toFixed(1)}
            yUnit="kg"
            height={200}
          />
        </div>
      </div>

      {/* Configurable Assumptions Form */}
      <form onSubmit={handleApplyAssumptions} style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '16px 20px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sliders size={16} color="var(--cyan-primary)" /> Configure Impact Rate Parameters
        </div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div className="form-group" style={{ flex: '1 1 180px' }}>
            <label className="form-label" htmlFor="tariff-input">
              Tariff Rate (₹ / kWh)
            </label>
            <input
              id="tariff-input"
              type="number"
              step="0.5"
              min="0"
              placeholder={`Current: ₹${assumptions.tariff_per_kwh}`}
              value={customTariff}
              onChange={(e) => setCustomTariff(e.target.value)}
              className="form-input"
            />
          </div>

          <div className="form-group" style={{ flex: '1 1 180px' }}>
            <label className="form-label" htmlFor="carbon-input">
              Carbon Intensity (kg CO₂e / kWh)
            </label>
            <input
              id="carbon-input"
              type="number"
              step="0.05"
              min="0"
              placeholder={`Current: ${assumptions.carbon_intensity_kg_per_kwh} kg/kWh`}
              value={customCarbon}
              onChange={(e) => setCustomCarbon(e.target.value)}
              className="form-input"
            />
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" className="btn btn-primary" disabled={isUpdating} style={{ padding: '9px 18px' }}>
              {isUpdating ? <RefreshCw size={14} className="spinner" /> : 'Apply Rates'}
            </button>
            <button type="button" onClick={handleResetAssumptions} className="btn btn-secondary" style={{ padding: '9px 14px' }}>
              Reset Defaults
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
