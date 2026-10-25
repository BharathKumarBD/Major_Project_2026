import React from 'react'
import { Activity, Zap, TrendingUp, IndianRupee, Cloud, ShieldCheck } from 'lucide-react'

export default function EnergyOverview({ stateData, forecastData, impactData }) {
  // Format helpers handling null/undefined strictly without fabricating 0
  const formatKwh = (val) => (typeof val === 'number' && !isNaN(val) ? `${val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kWh` : 'N/A')
  const formatCost = (val) => (typeof val === 'number' && !isNaN(val) ? `₹${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A')
  const formatCarbon = (val) => (typeof val === 'number' && !isNaN(val) ? `${val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg` : 'N/A')
  const formatPercent = (val) => (typeof val === 'number' && !isNaN(val) ? `${val > 0 ? '+' : ''}${(val * 100).toFixed(1)}%` : 'N/A')

  const actualKwh = stateData?.actual_kwh
  const expectedKwh = stateData?.expected_kwh
  const deviationRatio = stateData?.deviation_ratio
  const stateStatus = stateData?.state_status || 'UNAVAILABLE'

  // Forecast total from forecast or impact
  const forecastTotalKwh = forecastData?.summary?.total_kwh ?? stateData?.forecast_total_kwh
  const forecastTotalCost = impactData?.summary?.total_cost ?? stateData?.forecast_total_cost
  const forecastTotalCarbon = impactData?.summary?.total_carbon_kg ?? stateData?.forecast_total_carbon_kg

  // Status badge styling
  const getStatusBadge = (status) => {
    switch (status) {
      case 'NORMAL':
        return <span className="badge badge-normal"><ShieldCheck size={12} /> Normal</span>
      case 'WARNING':
        return <span className="badge badge-warning"><Activity size={12} /> Warning</span>
      case 'CRITICAL':
        return <span className="badge badge-critical"><Activity size={12} /> Critical</span>
      default:
        return <span className="badge badge-unavailable">Unavailable</span>
    }
  }

  return (
    <section aria-label="Energy Overview Metrics" style={{ marginBottom: 24 }}>
      <div className="kpi-grid">
        {/* 1. Current / Latest Actual Reading */}
        <div className="kpi-card" data-testid="kpi-actual">
          <div className="kpi-label">
            <Zap size={15} color="var(--cyan-primary)" /> Current Reading
          </div>
          <div className="kpi-value" style={{ color: 'var(--cyan-primary)' }}>
            {formatKwh(actualKwh)}
          </div>
          <div className="kpi-subtext">
            {deviationRatio !== null && deviationRatio !== undefined ? (
              <span style={{ color: deviationRatio > 0 ? 'var(--rose-primary)' : 'var(--emerald-primary)' }}>
                {formatPercent(deviationRatio)} vs baseline
              </span>
            ) : (
              'Latest recorded observation'
            )}
          </div>
        </div>

        {/* 2. Expected Energy Baseline */}
        <div className="kpi-card" data-testid="kpi-expected">
          <div className="kpi-label">
            <Activity size={15} color="#94a3b8" /> Expected Demand
          </div>
          <div className="kpi-value">
            {formatKwh(expectedKwh)}
          </div>
          <div className="kpi-subtext">LightGBM baseline model</div>
        </div>

        {/* 3. Twin State Status */}
        <div className="kpi-card" data-testid="kpi-status">
          <div className="kpi-label">
            <ShieldCheck size={15} color="var(--emerald-primary)" /> Twin State Status
          </div>
          <div className="kpi-value" style={{ fontSize: '1.45rem', paddingTop: 4 }}>
            {getStatusBadge(stateStatus)}
          </div>
          <div className="kpi-subtext">
            {stateData?.anomaly_type ? `Type: ${stateData.anomaly_type}` : 'Software state classification'}
          </div>
        </div>

        {/* 4. 24-Hour Forecast Cumulative */}
        <div className="kpi-card" data-testid="kpi-forecast">
          <div className="kpi-label">
            <TrendingUp size={15} color="#38bdf8" /> 24h Energy Forecast
          </div>
          <div className="kpi-value" style={{ color: '#38bdf8' }}>
            {formatKwh(forecastTotalKwh)}
          </div>
          <div className="kpi-subtext">Recursive LightGBM 24h sum</div>
        </div>

        {/* 5. Forecasted Cost */}
        <div className="kpi-card" data-testid="kpi-cost">
          <div className="kpi-label">
            <IndianRupee size={15} color="var(--amber-primary)" /> Forecasted Cost (24h)
          </div>
          <div className="kpi-value" style={{ color: 'var(--amber-primary)' }}>
            {formatCost(forecastTotalCost)}
          </div>
          <div className="kpi-subtext">
            @ {impactData?.assumptions?.tariff_per_kwh ? `₹${impactData.assumptions.tariff_per_kwh}/kWh` : 'Standard tariff'}
          </div>
        </div>

        {/* 6. Forecasted Carbon Emissions */}
        <div className="kpi-card" data-testid="kpi-carbon">
          <div className="kpi-label">
            <Cloud size={15} color="var(--emerald-primary)" /> Forecasted CO₂e (24h)
          </div>
          <div className="kpi-value" style={{ color: 'var(--emerald-primary)' }}>
            {formatCarbon(forecastTotalCarbon)}
          </div>
          <div className="kpi-subtext">
            @ {impactData?.assumptions?.carbon_intensity_kg_per_kwh ? `${impactData.assumptions.carbon_intensity_kg_per_kwh} kg/kWh` : 'Grid intensity'}
          </div>
        </div>
      </div>
    </section>
  )
}
