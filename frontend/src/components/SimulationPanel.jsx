import React, { useState } from 'react'
import { Play, Sparkles, AlertCircle, RefreshCw, CheckCircle2, TrendingUp, IndianRupee, Cloud } from 'lucide-react'
import LineChart from './charts/LineChart'
import ErrorMessage from './common/ErrorMessage'

export default function SimulationPanel({
  selectedBuilding,
  selectedMeter,
  onRunSimulation,
  simulationResult,
  isLoading = false,
  error = null,
}) {
  const [scenarioType, setScenarioType] = useState('ENERGY_MULTIPLIER')
  const [scenarioValue, setScenarioValue] = useState('1.10')
  const [validationError, setValidationError] = useState(null)

  // Quick preset buttons
  const handleScenarioChange = (newType) => {
    setScenarioType(newType)
    setValidationError(null)
    if (newType === 'ENERGY_MULTIPLIER') setScenarioValue('1.10')
    else if (newType === 'ENERGY_REDUCTION') setScenarioValue('0.15')
    else if (newType === 'FIXED_KWH_ADDITION') setScenarioValue('10.0')
  }

  // Client-side validation before submission
  const validateInput = () => {
    const val = parseFloat(scenarioValue)
    if (isNaN(val)) {
      return 'Please enter a valid numeric value.'
    }

    if (scenarioType === 'ENERGY_MULTIPLIER') {
      if (val <= 0) {
        return 'Multiplier factor must be greater than 0 (e.g. 1.10 for +10% demand).'
      }
      if (val > 5.0) {
        return 'Multiplier factor cannot exceed 5.0x for realistic modeling.'
      }
    } else if (scenarioType === 'ENERGY_REDUCTION') {
      if (val < 0 || val >= 1.0) {
        return 'Energy reduction fraction must be between 0.0 and 0.99 (e.g. 0.15 for 15% reduction).'
      }
    }

    return null
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const errorMsg = validateInput()
    if (errorMsg) {
      setValidationError(errorMsg)
      return
    }

    setValidationError(null)
    onRunSimulation({
      scenario_type: scenarioType,
      value: parseFloat(scenarioValue),
      meter: selectedMeter,
    })
  }

  // Format helpers
  const formatHour = (isoStr) => {
    if (!isoStr) return ''
    try {
      const d = new Date(isoStr)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
    } catch {
      return isoStr.substring(11, 16) || isoStr
    }
  }

  const chartSeries = [
    {
      key: 'baseline_kwh',
      name: 'Baseline Forecast',
      color: '#94a3b8',
      strokeWidth: 2,
      strokeDasharray: '4 4',
      areaFill: false,
    },
    {
      key: 'simulated_kwh',
      name: 'Simulated Demand',
      color: '#38bdf8',
      strokeWidth: 2.5,
      areaFill: true,
    },
  ]

  const summary = simulationResult?.summary
  const hourly = simulationResult?.hourly_simulation || []

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <Sparkles size={20} color="var(--cyan-primary)" /> What-If Energy Demand Simulation
          </div>
          <div className="section-caption">
            Hypothetical scenario modeling evaluated on top of the 24-hour forecast baseline without modifying original dataset or ML models.
          </div>
        </div>

        <span className="badge badge-info">Pure Simulation · Read-Only</span>
      </div>

      {/* Scenario Input Form */}
      <form
        onSubmit={handleSubmit}
        style={{
          background: 'rgba(15, 23, 42, 0.75)',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid var(--border-subtle)',
          marginBottom: 24,
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16, alignItems: 'flex-start' }}>
          {/* Scenario Type */}
          <div className="form-group">
            <label className="form-label" htmlFor="scenario-type-select">
              Select Scenario Type
            </label>
            <select
              id="scenario-type-select"
              className="form-select"
              value={scenarioType}
              onChange={(e) => handleScenarioChange(e.target.value)}
              disabled={isLoading}
              aria-label="Select Scenario Type"
            >
              <option value="ENERGY_MULTIPLIER">Energy Multiplier (Scale factor, e.g. 1.10 = +10%)</option>
              <option value="ENERGY_REDUCTION">Energy Reduction (Efficiency cut, e.g. 0.15 = 15%)</option>
              <option value="FIXED_KWH_ADDITION">Fixed Hourly Addition (Flat kWh shift, e.g. +10 kWh)</option>
            </select>
          </div>

          {/* Scenario Value */}
          <div className="form-group">
            <label className="form-label" htmlFor="scenario-value-input">
              {scenarioType === 'ENERGY_MULTIPLIER' && 'Multiplier Factor (e.g. 1.10 for +10% demand)'}
              {scenarioType === 'ENERGY_REDUCTION' && 'Reduction Fraction (e.g. 0.15 for 15% reduction)'}
              {scenarioType === 'FIXED_KWH_ADDITION' && 'Fixed kWh Addition per hour (e.g. +10.0 kWh)'}
            </label>
            <input
              id="scenario-value-input"
              type="number"
              step="any"
              className="form-input"
              value={scenarioValue}
              onChange={(e) => {
                setScenarioValue(e.target.value)
                setValidationError(null)
              }}
              disabled={isLoading}
              aria-label="Scenario parameter value"
            />
          </div>

          {/* Submit Button */}
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: '100%' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isLoading}
              style={{ width: '100%', height: '42px', marginTop: 22 }}
              data-testid="run-simulation-btn"
            >
              {isLoading ? (
                <>
                  <RefreshCw size={16} className="spinner" /> Simulating...
                </>
              ) : (
                <>
                  <Play size={16} fill="currentColor" /> Run What-If Simulation
                </>
              )}
            </button>
          </div>
        </div>

        {/* Client Validation Error */}
        {validationError && (
          <div style={{ marginTop: 12, fontSize: '0.82rem', color: 'var(--rose-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <AlertCircle size={15} /> {validationError}
          </div>
        )}
      </form>

      {/* Backend API Error Banner */}
      {error && <ErrorMessage title="Simulation Error" message={error} />}

      {/* Results Section */}
      {simulationResult && (
        <div style={{ marginTop: 20 }}>
          {/* Summary Delta Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
              gap: 14,
              marginBottom: 20,
            }}
          >
            {/* Energy Total Delta */}
            <div className="kpi-card" data-testid="sim-kpi-energy">
              <div className="kpi-label">
                <TrendingUp size={15} color="var(--cyan-primary)" /> Simulated Energy (24h)
              </div>
              <div className="kpi-value" style={{ color: 'var(--cyan-primary)' }}>
                {summary.simulated_total_kwh.toLocaleString(undefined, { maximumFractionDigits: 1 })} kWh
              </div>
              <div className="kpi-subtext" style={{ color: summary.delta_total_kwh > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)' }}>
                {summary.delta_total_kwh > 0 ? '+' : ''}{summary.delta_total_kwh.toFixed(1)} kWh ({summary.delta_total_percent > 0 ? '+' : ''}{summary.delta_total_percent.toFixed(1)}%)
              </div>
            </div>

            {/* Peak Demand */}
            <div className="kpi-card" data-testid="sim-kpi-peak">
              <div className="kpi-label">Simulated Peak Demand</div>
              <div className="kpi-value" style={{ color: 'var(--amber-primary)' }}>
                {summary.simulated_peak_kwh.toFixed(1)} kWh
              </div>
              <div className="kpi-subtext">
                Baseline Peak: {summary.baseline_peak_kwh.toFixed(1)} kWh
              </div>
            </div>

            {/* Cost Delta */}
            <div className="kpi-card" data-testid="sim-kpi-cost">
              <div className="kpi-label">
                <IndianRupee size={15} color="var(--amber-primary)" /> Simulated Cost (24h)
              </div>
              <div className="kpi-value" style={{ color: 'var(--amber-primary)' }}>
                ₹{summary.simulated_total_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className="kpi-subtext" style={{ color: summary.delta_total_cost > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)' }}>
                {summary.delta_total_cost > 0 ? '+' : ''}₹{summary.delta_total_cost.toFixed(2)}
              </div>
            </div>

            {/* Carbon Delta */}
            <div className="kpi-card" data-testid="sim-kpi-carbon">
              <div className="kpi-label">
                <Cloud size={15} color="var(--emerald-primary)" /> Simulated Carbon (24h)
              </div>
              <div className="kpi-value" style={{ color: 'var(--emerald-primary)' }}>
                {summary.simulated_total_carbon_kg.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg
              </div>
              <div className="kpi-subtext" style={{ color: summary.delta_total_carbon_kg > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)' }}>
                {summary.delta_total_carbon_kg > 0 ? '+' : ''}{summary.delta_total_carbon_kg.toFixed(1)} kg CO₂e
              </div>
            </div>
          </div>

          {/* Comparison Line Chart */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-subtle)', marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#f8fafc' }}>
                📈 Baseline vs. Simulated Hourly Trajectory
              </div>
              <div style={{ display: 'flex', gap: 16, fontSize: '0.8rem' }}>
                <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: '#94a3b8', borderStyle: 'dashed' }} />
                  Baseline Forecast
                </span>
                <span style={{ color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: '#38bdf8' }} />
                  Simulated Scenario
                </span>
              </div>
            </div>

            <LineChart
              data={hourly}
              series={chartSeries}
              xKey="timestamp"
              xFormatter={formatHour}
              yFormatter={(v) => v.toFixed(1)}
              yUnit="kWh"
              height={260}
            />
          </div>

          {/* Immutability confirmation notice */}
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', background: 'rgba(30, 41, 59, 0.4)', padding: '10px 14px', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <CheckCircle2 size={15} color="var(--emerald-primary)" />
            <span>
              <strong>Baseline Preserved:</strong> This simulation was executed on transient memory buffers. Baseline forecasts, historical records, and Twin State models remain completely unchanged.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
