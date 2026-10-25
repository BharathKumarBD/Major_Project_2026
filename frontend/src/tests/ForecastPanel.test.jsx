import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ForecastPanel from '../components/ForecastPanel'

describe('ForecastPanel Component', () => {
  const mockForecast = {
    building_id: 0,
    meter: 0,
    cutoff_time: '2016-12-30T23:00:00',
    weather_source: 'dataset_observed',
    summary: {
      total_kwh: 4646.5,
      peak_kwh: 230.1,
      peak_timestamp: '2016-12-31T00:00:00',
      min_kwh: 169.0,
      min_timestamp: '2016-12-31T14:00:00',
      average_kwh: 193.6,
    },
    hourly_forecast: [
      {
        step: 1,
        timestamp: '2016-12-31T00:00:00',
        predicted_kwh: 230.1,
        predicted_log: 5.44,
        air_temperature: 18.5,
      },
      {
        step: 2,
        timestamp: '2016-12-31T01:00:00',
        predicted_kwh: 216.7,
        predicted_log: 5.38,
        air_temperature: 18.0,
      },
    ],
  }

  it('renders forecast summary metrics and weather source', () => {
    render(<ForecastPanel forecastData={mockForecast} />)

    expect(screen.getByText(/4,646.5 kWh/i)).toBeInTheDocument()
    expect(screen.getByText(/230.1 kWh/i)).toBeInTheDocument()
    expect(screen.getByText(/dataset_observed/i)).toBeInTheDocument()
  })

  it('toggles between chart view and table view', () => {
    render(<ForecastPanel forecastData={mockForecast} />)

    // Initially in chart view
    const tableBtn = screen.getByRole('button', { name: /view forecast table/i })
    fireEvent.click(tableBtn)

    // Table view active
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByText('#1')).toBeInTheDocument()
    expect(screen.getByText('230.10')).toBeInTheDocument()
    expect(screen.getByText('18.5 °C')).toBeInTheDocument()

    // Switch back to chart view
    const chartBtn = screen.getByRole('button', { name: /view forecast chart/i })
    fireEvent.click(chartBtn)
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('renders empty state when forecast data is null', () => {
    render(<ForecastPanel forecastData={null} />)
    expect(screen.getByText(/forecast unavailable/i)).toBeInTheDocument()
  })
})
