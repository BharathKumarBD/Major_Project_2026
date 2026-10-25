import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  getHealth,
  getBuildings,
  getBuildingState,
  getBuildingForecast,
  getBuildingAnomalies,
  getBuildingImpact,
  getBuildingRecommendations,
  simulateBuildingDemand,
  ApiError,
} from '../services/api'

describe('API Service Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('getHealth calls /health endpoint', async () => {
    const mockHealth = { status: 'healthy', service: 'Twinergy 2.0 API', version: '2.0.0' }
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    })

    const res = await getHealth()
    expect(res).toEqual(mockHealth)
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/health'),
      expect.objectContaining({ method: 'GET' })
    )
  })

  it('getBuildings calls /buildings with optional meter filter', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ total_buildings: 2, building_ids: [0, 1], meters: [] }),
    })

    await getBuildings(0)
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/buildings?meter=0'),
      expect.anything()
    )
  })

  it('simulateBuildingDemand sends POST with JSON payload', async () => {
    const mockSim = {
      scenario_type: 'ENERGY_MULTIPLIER',
      scenario_value: 1.10,
      summary: {},
      hourly_simulation: [],
    }
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockSim,
    })

    const payload = { scenario_type: 'ENERGY_MULTIPLIER', value: 1.10, meter: 0 }
    const res = await simulateBuildingDemand(0, payload)
    expect(res).toEqual(mockSim)
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/buildings/0/simulate'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(payload),
      })
    )
  })

  it('throws ApiError with detail message on non-200 HTTP response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'Building ID 99999 not found.' }),
    })

    await expect(getBuildingState(99999)).rejects.toThrow('Building ID 99999 not found.')
  })

  it('propagates AbortError when request signal is aborted', async () => {
    const controller = new AbortController()
    controller.abort()

    global.fetch = vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError'))

    await expect(getBuildingForecast(0, 0, null, controller.signal)).rejects.toThrow('Aborted')
  })
})
