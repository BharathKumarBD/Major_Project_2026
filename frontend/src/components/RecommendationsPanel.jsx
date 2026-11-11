import React from 'react'
import { Lightbulb, Info, CheckCircle2, ShieldAlert, AlertTriangle } from 'lucide-react'
import EmptyState from './common/EmptyState'

export default function RecommendationsPanel({ recommendationsData }) {
  if (!recommendationsData || !recommendationsData.recommendations) {
    return (
      <div className="glass-card">
        <div className="section-header">
          <div className="section-title">
            <Lightbulb size={20} /> Energy Management Recommendations
          </div>
        </div>
        <EmptyState title="No Recommendations Available" message="Recommendations could not be generated for this building state." />
      </div>
    )
  }

  const { recommendations = [], total_count = 0, twin_state_status, generated_at } = recommendationsData

  const formatDate = (isoStr) => {
    if (!isoStr) return ''
    try {
      const d = new Date(isoStr)
      return `${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })} · ${d.toLocaleDateString([], { day: 'numeric', month: 'short' })}`
    } catch {
      return isoStr
    }
  }

  const cleanEvidence = (text) => {
    if (!text || typeof text !== 'string') return text
    return text.replace(/â[‚\u201a]?¹/g, '₹')
  }

  const getPriorityBadge = (priority) => {
    const p = String(priority).toUpperCase()
    switch (p) {
      case 'CRITICAL':
        return { badgeClass: 'badge-critical', icon: <AlertTriangle size={13} />, color: '#f43f5e', border: '#e11d48' }
      case 'HIGH':
        return { badgeClass: 'badge-warning', icon: <AlertTriangle size={13} />, color: '#fbbf24', border: '#d97706' }
      case 'MEDIUM':
        return { badgeClass: 'badge-warning', icon: <Info size={13} />, color: '#f59e0b', border: '#b45309' }
      case 'LOW':
        return { badgeClass: 'badge-normal', icon: <CheckCircle2 size={13} />, color: '#34d399', border: '#059669' }
      default:
        return { badgeClass: 'badge-info', icon: <Info size={13} />, color: '#38bdf8', border: '#0284c7' }
    }
  }

  return (
    <div className="glass-card" style={{ marginBottom: 24 }}>
      {/* Header */}
      <div className="section-header">
        <div>
          <div className="section-title">
            <Lightbulb size={20} /> Energy Management Recommendations
          </div>
          <div className="section-caption">
            Rule-based, explainable advisory recommendations sorted strictly by priority. Generated from TwinState conditions, anomaly signals, and forecasted demand.
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="badge badge-info">{total_count} Actions Identified</span>
        </div>
      </div>

      {/* Advisory Non-Causal Notice */}
      <div className="alert-banner alert-info" style={{ marginBottom: 20 }}>
        <Info size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div style={{ flex: 1, fontSize: '0.82rem' }}>
          <strong>Advisory Guidelines:</strong> Recommendations are deterministic suggestions intended for human building operators. They do not claim physical causality or execute automated HVAC controls.
        </div>
      </div>

      {/* Recommendations List */}
      {recommendations.length === 0 ? (
        <EmptyState
          title="No Active Recommendations"
          message="All energy, cost, and carbon parameters are within standard baseline thresholds."
        />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {recommendations.map((rec) => {
            const pStyle = getPriorityBadge(rec.priority)
            return (
              <div
                key={rec.recommendation_id}
                style={{
                  background: 'rgba(15, 23, 42, 0.75)',
                  border: '1px solid var(--border-subtle)',
                  borderLeft: `4px solid ${pStyle.border}`,
                  borderRadius: '10px',
                  padding: '16px 20px',
                  transition: 'all 0.2s ease',
                }}
              >
                {/* Top Meta Line: ID, Title, Priority, Category */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#94a3b8', background: 'rgba(30, 41, 59, 0.8)', padding: '2px 8px', borderRadius: '4px' }}>
                      {rec.recommendation_id}
                    </span>
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>
                      {rec.title}
                    </h3>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge ${pStyle.badgeClass}`}>
                      {pStyle.icon} {rec.priority}
                    </span>
                    <span
                      style={{
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        color: '#94a3b8',
                        background: 'rgba(30, 41, 59, 0.6)',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        textTransform: 'uppercase',
                      }}
                    >
                      {rec.category}
                    </span>
                  </div>
                </div>

                {/* Suggested Action Item */}
                <div style={{ fontSize: '0.88rem', color: '#f1f5f9', fontWeight: 600, marginBottom: 6 }}>
                  👉 <span style={{ color: 'var(--cyan-primary)' }}>Recommended Action:</span> {rec.action}
                </div>

                {/* Trigger Reason */}
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1', marginBottom: 6 }}>
                  <strong>Trigger Reason:</strong> {rec.reason}
                </div>

                {/* Quantitative Evidence */}
                {rec.evidence && (
                  <div
                    style={{
                      fontSize: '0.78rem',
                      color: '#94a3b8',
                      fontStyle: 'italic',
                      background: 'rgba(0, 0, 0, 0.25)',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      marginTop: 6,
                    }}
                  >
                    <strong>Data Evidence:</strong> {cleanEvidence(rec.evidence)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
