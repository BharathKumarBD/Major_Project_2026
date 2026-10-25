import React from 'react'
import { Building2, Zap, Wifi, WifiOff, AlertTriangle } from 'lucide-react'

export default function Header({
  apiStatus,
  buildings = [],
  meters = [],
  selectedBuilding,
  selectedMeter,
  onSelectBuilding,
  onSelectMeter,
  isLoadingBuildings = false,
}) {
  const currentMeterInfo = meters.find((m) => m.meter_id === selectedMeter) || {
    name: selectedMeter === 0 ? 'Electricity' : `Meter ${selectedMeter}`,
  }

  return (
    <header
      style={{
        background: 'rgba(11, 17, 32, 0.95)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border-subtle)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        padding: '16px 24px',
      }}
    >
      <div
        style={{
          maxWidth: '1440px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 16,
        }}
      >
        {/* Brand & Identity */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #0284c7, #38bdf8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 16px rgba(56, 189, 248, 0.4)',
            }}
          >
            <Zap size={24} color="#060913" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.03em' }}>
                Twinergy <span style={{ color: 'var(--cyan-primary)' }}>2.0</span>
              </h1>
              <span
                style={{
                  background: 'rgba(56, 189, 248, 0.12)',
                  color: 'var(--cyan-primary)',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: '6px',
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  textTransform: 'uppercase',
                }}
              >
                Digital Twin
              </span>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              AI Energy Forecasting, Anomaly Analytics & What-If Simulation
            </div>
          </div>
        </div>

        {/* Controls: Building & Meter Selectors + Connection Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
          {/* Building Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label htmlFor="building-select" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8' }}>
              Building:
            </label>
            <select
              id="building-select"
              className="form-select"
              value={selectedBuilding}
              onChange={(e) => onSelectBuilding(Number(e.target.value))}
              disabled={isLoadingBuildings || buildings.length === 0}
              style={{ minWidth: 140, fontWeight: 600 }}
              aria-label="Select Building ID"
            >
              {isLoadingBuildings ? (
                <option value={selectedBuilding}>Loading...</option>
              ) : (
                buildings.map((id) => (
                  <option key={`b-${id}`} value={id}>
                    Building #{id}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Meter Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label htmlFor="meter-select" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8' }}>
              Meter:
            </label>
            <select
              id="meter-select"
              className="form-select"
              value={selectedMeter}
              onChange={(e) => onSelectMeter(Number(e.target.value))}
              style={{ minWidth: 150, fontWeight: 600 }}
              aria-label="Select Meter Type"
            >
              {meters.map((m) => (
                <option key={`m-${m.meter_id}`} value={m.meter_id}>
                  {m.name} ({m.meter_id})
                </option>
              ))}
            </select>
          </div>

          {/* Active Target Pill */}
          <div
            className="badge badge-info"
            style={{ padding: '6px 12px', fontSize: '0.8rem', textTransform: 'none' }}
          >
            <Building2 size={14} />
            Building #{selectedBuilding} · {currentMeterInfo.name}
          </div>

          {/* API Health Pill */}
          <div
            className={`badge ${apiStatus === 'connected' ? 'badge-normal' : 'badge-critical'}`}
            style={{ padding: '6px 12px', fontSize: '0.8rem', textTransform: 'none' }}
            title={apiStatus === 'connected' ? 'API backend connected' : 'Cannot connect to API backend'}
          >
            {apiStatus === 'connected' ? (
              <>
                <Wifi size={14} /> API Online
              </>
            ) : (
              <>
                <WifiOff size={14} /> API Offline
              </>
            )}
          </div>
        </div>
      </div>

      {/* Global Historical / Model-Derived Advisory Banner */}
      <div
        style={{
          maxWidth: '1440px',
          margin: '12px auto 0 auto',
          background: 'rgba(30, 41, 59, 0.45)',
          border: '1px solid rgba(51, 65, 85, 0.4)',
          borderRadius: '8px',
          padding: '6px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: '0.78rem',
          color: '#cbd5e1',
        }}
      >
        <AlertTriangle size={14} color="#fbbf24" style={{ flexShrink: 0 }} />
        <span>
          <strong>Software Digital Twin:</strong> Visualizing historical ASHRAE benchmark observations & ML model predictions. Not connected to live IoT sensor feeds or physical actuators.
        </span>
      </div>
    </header>
  )
}
