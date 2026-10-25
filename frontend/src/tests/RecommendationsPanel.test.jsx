import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import RecommendationsPanel from '../components/RecommendationsPanel'

describe('RecommendationsPanel Component', () => {
  const mockRecs = {
    building_id: 0,
    meter: 0,
    twin_state_status: 'WARNING',
    total_count: 3,
    generated_at: '2026-10-25T14:00:00Z',
    recommendations: [
      {
        recommendation_id: 'REC-ANOM-001',
        priority: 'CRITICAL',
        category: 'ANOMALY',
        title: 'Investigate Critical Consumption Spike',
        action: 'Review electrical sub-circuits immediately.',
        reason: 'Consumption surged 60% above expected baseline.',
        evidence: 'Actual consumption 200 kWh vs 125 kWh expected.',
      },
      {
        recommendation_id: 'REC-COST-002',
        priority: 'HIGH',
        category: 'COST',
        title: 'Shift Flexible Loads Off-Peak',
        action: 'Schedule water pumping during off-peak hours.',
        reason: 'Projected 24h cost exceeds threshold.',
        evidence: 'Projected cost is ₹25,000.',
      },
      {
        recommendation_id: 'REC-OPER-003',
        priority: 'LOW',
        category: 'OPERATIONS',
        title: 'Inspect Temperature Setpoints',
        action: 'Verify chiller thermostat schedule.',
        reason: 'Standard periodic efficiency review.',
        evidence: 'Thermostat set 2 degrees below optimal.',
      },
    ],
  }

  it('renders recommendations in the exact priority order returned by API', () => {
    render(<RecommendationsPanel recommendationsData={mockRecs} />)

    const titles = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(titles).toEqual([
      'Investigate Critical Consumption Spike',
      'Shift Flexible Loads Off-Peak',
      'Inspect Temperature Setpoints',
    ])

    // Verify priority badges
    expect(screen.getByText('CRITICAL')).toBeInTheDocument()
    expect(screen.getByText('HIGH')).toBeInTheDocument()
    expect(screen.getByText('LOW')).toBeInTheDocument()

    // Verify actions and evidence
    expect(screen.getByText(/Review electrical sub-circuits immediately/i)).toBeInTheDocument()
    expect(screen.getByText(/Actual consumption 200 kWh vs 125 kWh expected/i)).toBeInTheDocument()
  })

  it('displays empty state when recommendations array is empty', () => {
    const emptyRecs = {
      building_id: 0,
      meter: 0,
      total_count: 0,
      recommendations: [],
    }

    render(<RecommendationsPanel recommendationsData={emptyRecs} />)
    expect(screen.getByText(/no active recommendations/i)).toBeInTheDocument()
  })
})
