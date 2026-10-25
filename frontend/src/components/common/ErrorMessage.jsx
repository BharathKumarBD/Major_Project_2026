import React from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

export default function ErrorMessage({ title = 'Request Failed', message, onRetry }) {
  return (
    <div className="alert-banner alert-error" role="alert">
      <AlertCircle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
      <div style={{ flex: 1 }}>
        <strong style={{ display: 'block', marginBottom: 4 }}>{title}</strong>
        <div>{message || 'An unexpected error occurred while communicating with the Twinergy API.'}</div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="btn btn-secondary"
            style={{ marginTop: 10, padding: '6px 14px', fontSize: '0.8rem' }}
          >
            <RefreshCw size={14} /> Retry
          </button>
        )}
      </div>
    </div>
  )
}
