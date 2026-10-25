import React from 'react'
import { Inbox } from 'lucide-react'

export default function EmptyState({ title = 'No Data Available', message = 'No records found for the selected building and meter.' }) {
  return (
    <div className="loading-box" style={{ padding: '40px 20px' }}>
      <Inbox size={42} style={{ color: '#475569' }} />
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontWeight: 600, color: '#f1f5f9', marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{message}</div>
      </div>
    </div>
  )
}
