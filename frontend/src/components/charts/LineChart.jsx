import React, { useState } from 'react'

/**
 * Premium responsive SVG Line Chart with multi-series, gradient area fills,
 * dashed/solid line support, peak highlights, and interactive hover tooltips.
 */
export default function LineChart({
  data = [],
  series = [],
  xKey = 'timestamp',
  xFormatter = (val) => val,
  yFormatter = (val) => val.toFixed(1),
  yUnit = 'kWh',
  height = 300,
  highlightPeak = false,
  peakKey = null,
}) {
  const [hoverIndex, setHoverIndex] = useState(null)

  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
        No chart data available
      </div>
    )
  }

  // Padding
  const padding = { top: 20, right: 25, bottom: 40, left: 55 }
  const width = 800 // Virtual coordinate width

  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom

  // Compute min/max for Y axis across all active series
  let minY = Infinity
  let maxY = -Infinity

  series.forEach((s) => {
    data.forEach((d) => {
      const val = d[s.key]
      if (typeof val === 'number' && !isNaN(val)) {
        if (val < minY) minY = val
        if (val > maxY) maxY = val
      }
    })
  })

  if (minY === Infinity || maxY === -Infinity) {
    minY = 0
    maxY = 100
  }

  // Add 10% breathing room to Y axis
  const ySpan = maxY - minY || 1
  const effectiveMinY = Math.max(0, minY - ySpan * 0.05)
  const effectiveMaxY = maxY + ySpan * 0.12
  const rangeY = effectiveMaxY - effectiveMinY || 1

  // Coordinate scales
  const getX = (idx) => padding.left + (idx / Math.max(data.length - 1, 1)) * plotWidth
  const getY = (val) => padding.top + plotHeight - ((val - effectiveMinY) / rangeY) * plotHeight

  // Y Axis ticks (4 horizontal grid lines)
  const yTicks = [0, 0.33, 0.66, 1].map((ratio) => {
    const val = effectiveMinY + ratio * rangeY
    const yCoord = padding.top + plotHeight - ratio * plotHeight
    return { val, yCoord }
  })

  // X Axis ticks: sample ~5 to 8 timestamps
  const step = Math.max(1, Math.floor(data.length / 6))
  const xTicks = data
    .map((d, idx) => ({ d, idx }))
    .filter(({ idx }) => idx % step === 0 || idx === data.length - 1)

  // Find peak index if requested
  let peakIndex = -1
  if (highlightPeak && peakKey) {
    let maxVal = -Infinity
    data.forEach((d, idx) => {
      const v = d[peakKey]
      if (typeof v === 'number' && v > maxVal) {
        maxVal = v
        peakIndex = idx
      }
    })
  }

  return (
    <div style={{ position: 'relative', width: '100%', userSelect: 'none' }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block', overflow: 'visible' }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <defs>
          {series.map((s) => (
            <linearGradient key={`grad-${s.key}`} id={`gradient-${s.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0.0" />
            </linearGradient>
          ))}
        </defs>

        {/* Horizontal Gridlines & Y Labels */}
        {yTicks.map(({ val, yCoord }, idx) => (
          <g key={`ytick-${idx}`}>
            <line
              x1={padding.left}
              y1={yCoord}
              x2={width - padding.right}
              y2={yCoord}
              stroke="rgba(51, 65, 85, 0.4)"
              strokeDasharray={idx === 0 ? 'none' : '3 3'}
            />
            <text
              x={padding.left - 8}
              y={yCoord + 4}
              textAnchor="end"
              fill="#64748b"
              fontSize="10"
              fontFamily="var(--font-mono)"
            >
              {yFormatter(val)}
            </text>
          </g>
        ))}

        {/* X Axis Ticks & Labels */}
        {xTicks.map(({ d, idx }) => {
          const xCoord = getX(idx)
          return (
            <g key={`xtick-${idx}`}>
              <line
                x1={xCoord}
                y1={padding.top + plotHeight}
                x2={xCoord}
                y2={padding.top + plotHeight + 5}
                stroke="rgba(51, 65, 85, 0.6)"
              />
              <text
                x={xCoord}
                y={padding.top + plotHeight + 18}
                textAnchor="middle"
                fill="#64748b"
                fontSize="10"
                fontFamily="var(--font-mono)"
              >
                {xFormatter(d[xKey])}
              </text>
            </g>
          )
        })}

        {/* Series Area Fills */}
        {series
          .filter((s) => s.areaFill)
          .map((s) => {
            const points = data.map((d, i) => `${getX(i)},${getY(d[s.key] || 0)}`)
            const areaPath = `M ${getX(0)},${padding.top + plotHeight} L ${points.join(' L ')} L ${getX(data.length - 1)},${padding.top + plotHeight} Z`
            return <path key={`area-${s.key}`} d={areaPath} fill={`url(#gradient-${s.key})`} />
          })}

        {/* Series Lines */}
        {series.map((s) => {
          const points = data
            .filter((d) => typeof d[s.key] === 'number')
            .map((d, i) => `${getX(i)},${getY(d[s.key])}`)
          const linePath = `M ${points.join(' L ')}`

          return (
            <path
              key={`line-${s.key}`}
              d={linePath}
              fill="none"
              stroke={s.color}
              strokeWidth={s.strokeWidth || 2.2}
              strokeDasharray={s.strokeDasharray || 'none'}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )
        })}

        {/* Peak Indicator Dot */}
        {peakIndex >= 0 && (
          <g>
            <circle
              cx={getX(peakIndex)}
              cy={getY(data[peakIndex][peakKey])}
              r="6"
              fill="#fbbf24"
              stroke="#0f172a"
              strokeWidth="2"
            />
            <circle
              cx={getX(peakIndex)}
              cy={getY(data[peakIndex][peakKey])}
              r="10"
              fill="none"
              stroke="#fbbf24"
              strokeWidth="1.5"
              opacity="0.6"
            />
          </g>
        )}

        {/* Hover Crosshair */}
        {hoverIndex !== null && (
          <g>
            <line
              x1={getX(hoverIndex)}
              y1={padding.top}
              x2={getX(hoverIndex)}
              y2={padding.top + plotHeight}
              stroke="rgba(56, 189, 248, 0.6)"
              strokeWidth="1.5"
              strokeDasharray="3 3"
            />
            {series.map((s) => {
              const val = data[hoverIndex][s.key]
              if (typeof val !== 'number') return null
              return (
                <circle
                  key={`hover-dot-${s.key}`}
                  cx={getX(hoverIndex)}
                  cy={getY(val)}
                  r="5"
                  fill={s.color}
                  stroke="#0f172a"
                  strokeWidth="2"
                />
              )
            })}
          </g>
        )}

        {/* Invisible Hover Hitboxes across X steps */}
        {data.map((_, idx) => {
          const xCenter = getX(idx)
          const sliceWidth = plotWidth / data.length
          return (
            <rect
              key={`hitbox-${idx}`}
              x={xCenter - sliceWidth / 2}
              y={padding.top}
              width={sliceWidth}
              height={plotHeight}
              fill="transparent"
              style={{ cursor: 'crosshair' }}
              onMouseEnter={() => setHoverIndex(idx)}
            />
          )
        })}
      </svg>

      {/* Floating Tooltip Box */}
      {hoverIndex !== null && data[hoverIndex] && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: Math.min(Math.max(10, (getX(hoverIndex) / width) * 100 - 15), 70) + '%',
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid #38bdf8',
            borderRadius: '8px',
            padding: '8px 12px',
            pointerEvents: 'none',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.6)',
            fontSize: '0.78rem',
            zIndex: 10,
            minWidth: 160,
          }}
        >
          <div style={{ color: '#94a3b8', marginBottom: 4, fontWeight: 600 }}>
            {xFormatter(data[hoverIndex][xKey])}
          </div>
          {series.map((s) => {
            const val = data[hoverIndex][s.key]
            if (typeof val !== 'number') return null
            return (
              <div
                key={`tt-${s.key}`}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: 12,
                  color: s.color,
                  fontWeight: 600,
                  marginTop: 2,
                }}
              >
                <span>{s.name}:</span>
                <span style={{ color: '#ffffff' }}>
                  {yFormatter(val)} {yUnit}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
