import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import EnergyOverview from '../components/EnergyOverview'

describe('EnergyOverview Component', () => {
  const mockState = {
    actual_kwh: 175.4,
    expected_kwh: 180.7,
    deviation_ratio: -0.029,
    state_status: 'NORMAL',
    anomaly_type: 'NORMAL',
    forecast_total_kwh: 4646.5,
    forecast_total_cost: 37171.88,
    forecast_total_carbon_kg: 3252.5,
  }

  const mockForecast = {
    summary: {
      total_kwh: 4646.5,
      peak_kwh: 230.1,
    },
  }

  const mockImpact = {
    summary: {
      total_cost: 37171.88,
      total_carbon_kg: 3252.5,
    },
    assumptions: {
      tariff_per_kwh: 8.0,
      carbon_intensity_kg_per_kwh: 0.70,
    },
  }

  it('renders all metrics with proper formatting when data is provided', () => {
    render(
      <EnergyOverview
        stateData={mockState}
        forecastData={mockForecast}
        impactData={mockImpact}
      />
    )

    // Current reading
    expect(screen.getByTestId('kpi-actual')).toHaveTextContent('175.4 kWh')
    // Expected demand
    expect(screen.getByTestId('kpi-expected')).toHaveTextContent('180.7 kWh')
    // Status
    expect(screen.getByTestId('kpi-status')).toHaveTextContent('Normal')
    // Forecast 24h
    expect(screen.getByTestId('kpi-forecast')).toHaveTextContent('4,646.5 kWh')
    // Cost
    expect(screen.getByTestId('kpi-cost')).toHaveTextContent('₹37,171.88')
    // Carbon
    expect(screen.getByTestId('kpi-carbon')).toHaveTextContent('3,252.5 kg')
  })

  it('displays N/A for missing or null metrics and never fabricates zeros', () => {
    const emptyState = {
      actual_kwh: null,
      expected_kwh: null,
      deviation_ratio: null,
      state_status: 'UNAVAILABLE',
      forecast_total_kwh: null,
      forecast_total_cost: null,
      forecast_total_carbon_kg: null,
    }

    render(
      <EnergyOverview
        stateData={emptyState}
        forecastData={null}
        impactData={null}
      />
    )

    // Must show N/A, NOT 0.0 or fabricated numbers
    expect(screen.getByTestId('kpi-actual')).toHaveTextContent('N/A')
    expect(screen.getByTestId('kpi-expected')).toHaveTextContent('N/A')
    expect(screen.getByTestId('kpi-forecast')).toHaveTextContent('N/A')
    expect(screen.getByTestId('kpi-cost')).toHaveTextContent('N/A')
    expect(screen.getByTestId('kpi-carbon')).toHaveTextContent('N/A')
    expect(screen.getByTestId('kpi-status')).toHaveTextContent('Unavailable')
  })
})
