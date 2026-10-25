import React from 'react'

export default function LoadingSpinner({ text = 'Loading energy intelligence...' }) {
  return (
    <div className="loading-box" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <span style={{ fontSize: '0.9rem', color: '#94a3b8', fontWeight: 500 }}>{text}</span>
    </div>
  )
}
