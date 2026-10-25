import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Header from '../components/Header'

describe('Header Component', () => {
  const mockBuildings = [0, 1, 2, 5]
  const mockMeters = [
    { meter_id: 0, name: 'Electricity' },
    { meter_id: 1, name: 'Chilled Water' },
  ]

  it('populates building and meter selectors from API data', () => {
    render(
      <Header
        apiStatus="connected"
        buildings={mockBuildings}
        meters={mockMeters}
        selectedBuilding={0}
        selectedMeter={0}
        onSelectBuilding={vi.fn()}
        onSelectMeter={vi.fn()}
      />
    )

    // Verify building options
    const buildingSelect = screen.getByLabelText(/select building id/i)
    expect(buildingSelect).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Building #0' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Building #1' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Building #5' })).toBeInTheDocument()

    // Verify meter options
    const meterSelect = screen.getByLabelText(/select meter type/i)
    expect(meterSelect).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Electricity (0)' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Chilled Water (1)' })).toBeInTheDocument()

    // Verify API Online badge
    expect(screen.getByText(/api online/i)).toBeInTheDocument()
  })

  it('calls onSelectBuilding when user selects a different building', () => {
    const handleSelectBuilding = vi.fn()
    render(
      <Header
        apiStatus="connected"
        buildings={mockBuildings}
        meters={mockMeters}
        selectedBuilding={0}
        selectedMeter={0}
        onSelectBuilding={handleSelectBuilding}
        onSelectMeter={vi.fn()}
      />
    )

    const buildingSelect = screen.getByLabelText(/select building id/i)
    fireEvent.change(buildingSelect, { target: { value: '2' } })
    expect(handleSelectBuilding).toHaveBeenCalledWith(2)
  })

  it('displays API Offline badge when API connection fails', () => {
    render(
      <Header
        apiStatus="disconnected"
        buildings={[]}
        meters={mockMeters}
        selectedBuilding={0}
        selectedMeter={0}
        onSelectBuilding={vi.fn()}
        onSelectMeter={vi.fn()}
      />
    )

    expect(screen.getByText(/api offline/i)).toBeInTheDocument()
  })
})
