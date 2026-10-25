import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import SimulationPanel from '../components/SimulationPanel'

describe('SimulationPanel Component', () => {
  it('supports ENERGY_MULTIPLIER and sends correct payload', () => {
    const handleRunSimulation = vi.fn()
    render(
      <SimulationPanel
        selectedBuilding={0}
        selectedMeter={0}
        onRunSimulation={handleRunSimulation}
        simulationResult={null}
      />
    )

    // Select ENERGY_MULTIPLIER (default) and change value to 1.25
    const valueInput = screen.getByLabelText(/scenario parameter value/i)
    fireEvent.change(valueInput, { target: { value: '1.25' } })

    const submitBtn = screen.getByTestId('run-simulation-btn')
    fireEvent.click(submitBtn)

    expect(handleRunSimulation).toHaveBeenCalledWith({
      scenario_type: 'ENERGY_MULTIPLIER',
      value: 1.25,
      meter: 0,
    })
  })

  it('supports ENERGY_REDUCTION and sends correct payload', () => {
    const handleRunSimulation = vi.fn()
    render(
      <SimulationPanel
        selectedBuilding={0}
        selectedMeter={0}
        onRunSimulation={handleRunSimulation}
        simulationResult={null}
      />
    )

    // Select ENERGY_REDUCTION
    const typeSelect = screen.getByLabelText(/select scenario type/i)
    fireEvent.change(typeSelect, { target: { value: 'ENERGY_REDUCTION' } })

    const valueInput = screen.getByLabelText(/scenario parameter value/i)
    fireEvent.change(valueInput, { target: { value: '0.20' } })

    const submitBtn = screen.getByTestId('run-simulation-btn')
    fireEvent.click(submitBtn)

    expect(handleRunSimulation).toHaveBeenCalledWith({
      scenario_type: 'ENERGY_REDUCTION',
      value: 0.20,
      meter: 0,
    })
  })

  it('supports FIXED_KWH_ADDITION and sends correct payload', () => {
    const handleRunSimulation = vi.fn()
    render(
      <SimulationPanel
        selectedBuilding={1}
        selectedMeter={0}
        onRunSimulation={handleRunSimulation}
        simulationResult={null}
      />
    )

    // Select FIXED_KWH_ADDITION
    const typeSelect = screen.getByLabelText(/select scenario type/i)
    fireEvent.change(typeSelect, { target: { value: 'FIXED_KWH_ADDITION' } })

    const valueInput = screen.getByLabelText(/scenario parameter value/i)
    fireEvent.change(valueInput, { target: { value: '15.5' } })

    const submitBtn = screen.getByTestId('run-simulation-btn')
    fireEvent.click(submitBtn)

    expect(handleRunSimulation).toHaveBeenCalledWith({
      scenario_type: 'FIXED_KWH_ADDITION',
      value: 15.5,
      meter: 0,
    })
  })

  it('rejects non-positive ENERGY_MULTIPLIER before submission', () => {
    const handleRunSimulation = vi.fn()
    render(
      <SimulationPanel
        selectedBuilding={0}
        selectedMeter={0}
        onRunSimulation={handleRunSimulation}
        simulationResult={null}
      />
    )

    const valueInput = screen.getByLabelText(/scenario parameter value/i)
    fireEvent.change(valueInput, { target: { value: '-0.5' } })

    const submitBtn = screen.getByTestId('run-simulation-btn')
    fireEvent.click(submitBtn)

    // Should NOT call API
    expect(handleRunSimulation).not.toHaveBeenCalled()
    // Should show client validation error
    expect(screen.getByText(/multiplier factor must be greater than 0/i)).toBeInTheDocument()
  })

  it('rejects out-of-bounds ENERGY_REDUCTION (>= 1.0) before submission', () => {
    const handleRunSimulation = vi.fn()
    render(
      <SimulationPanel
        selectedBuilding={0}
        selectedMeter={0}
        onRunSimulation={handleRunSimulation}
        simulationResult={null}
      />
    )

    const typeSelect = screen.getByLabelText(/select scenario type/i)
    fireEvent.change(typeSelect, { target: { value: 'ENERGY_REDUCTION' } })

    const valueInput = screen.getByLabelText(/scenario parameter value/i)
    fireEvent.change(valueInput, { target: { value: '1.20' } })

    const submitBtn = screen.getByTestId('run-simulation-btn')
    fireEvent.click(submitBtn)

    expect(handleRunSimulation).not.toHaveBeenCalled()
    expect(screen.getByText(/reduction fraction must be between 0.0 and 0.99/i)).toBeInTheDocument()
  })

  it('renders simulation results and clearly notes baseline is preserved', () => {
    const mockResult = {
      scenario_type: 'ENERGY_MULTIPLIER',
      scenario_value: 1.10,
      summary: {
        baseline_total_kwh: 4000.0,
        simulated_total_kwh: 4400.0,
        delta_total_kwh: 400.0,
        delta_total_percent: 10.0,
        baseline_peak_kwh: 200.0,
        simulated_peak_kwh: 220.0,
        baseline_total_cost: 32000.0,
        simulated_total_cost: 35200.0,
        delta_total_cost: 3200.0,
        baseline_total_carbon_kg: 2800.0,
        simulated_total_carbon_kg: 3080.0,
        delta_total_carbon_kg: 280.0,
      },
      hourly_simulation: [
        {
          timestamp: '2016-12-31T00:00:00',
          baseline_kwh: 200.0,
          simulated_kwh: 220.0,
          delta_kwh: 20.0,
          delta_percent: 10.0,
        },
      ],
    }

    render(
      <SimulationPanel
        selectedBuilding={0}
        selectedMeter={0}
        onRunSimulation={vi.fn()}
        simulationResult={mockResult}
      />
    )

    // Verify simulated metrics
    expect(screen.getByTestId('sim-kpi-energy')).toHaveTextContent('4,400 kWh')
    expect(screen.getByTestId('sim-kpi-peak')).toHaveTextContent('220.0 kWh')
    expect(screen.getByTestId('sim-kpi-cost')).toHaveTextContent('₹35,200.00')
    expect(screen.getByTestId('sim-kpi-carbon')).toHaveTextContent('3,080.0 kg')

    // Verify baseline preserved notice
    expect(screen.getByText(/baseline preserved/i)).toBeInTheDocument()
  })
})
