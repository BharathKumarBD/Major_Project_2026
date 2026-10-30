/**
 * Centralized API Client for Twinergy REST API.
 *
 * Consumes the FastAPI backend endpoints with:
 * - Environment-configured base URL (VITE_API_BASE_URL with fallback to http://127.0.0.1:8000)
 * - AbortController signal support for stale request cancellation on rapid UI switching
 * - Normalized error handling extracting backend validation details
 * - Zero data fabrication on failure
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

/**
 * Custom API Error wrapping HTTP status, code, and backend detail.
 */
export class ApiError extends Error {
  constructor(message, status = 500, detail = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/**
 * Internal generic request handler with signal support and error parsing.
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  const config = {
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers,
    },
    ...options,
  }

  let response
  try {
    response = await fetch(url, config)
  } catch (err) {
    if (err.name === 'AbortError') {
      throw err // Re-throw AbortError so caller can silently ignore aborted requests
    }
    throw new ApiError(
      `Network connection failed. Could not reach Twinergy API at ${API_BASE_URL}. Ensure the backend is running.`,
      0,
      err.message
    )
  }

  if (!response.ok) {
    let errorDetail = null
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`

    try {
      const errorJson = await response.json()
      errorDetail = errorJson.detail || errorJson
      if (typeof errorDetail === 'string') {
        errorMessage = errorDetail
      } else if (Array.isArray(errorDetail)) {
        // Pydantic validation errors list
        errorMessage = errorDetail.map((e) => e.msg || JSON.stringify(e)).join('; ')
      } else if (errorDetail && typeof errorDetail === 'object') {
        errorMessage = errorDetail.message || JSON.stringify(errorDetail)
      }
    } catch {
      // Body not JSON
    }

    throw new ApiError(errorMessage, response.status, errorDetail)
  }

  return response.json()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Health
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check service health status.
 * @param {AbortSignal} [signal]
 * @returns {Promise<{ status: string, service: string, version: string }>}
 */
export async function getHealth(signal) {
  return request('/health', { method: 'GET', signal })
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Buildings Listing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve available building IDs and meter types.
 * @param {number} [meter] - Optional meter ID filter
 * @param {AbortSignal} [signal]
 * @returns {Promise<{ total_buildings: number, building_ids: number[], meters: Array<{meter_id: number, name: string}> }>}
 */
export async function getBuildings(meter = null, signal) {
  const query = meter !== null && meter !== undefined ? `?meter=${encodeURIComponent(meter)}` : ''
  return request(`/buildings${query}`, { method: 'GET', signal })
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Building Digital Twin State
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve software TwinState for a building and meter.
 * @param {number} buildingId
 * @param {number} [meter=0]
 * @param {string} [refTime] - Optional ISO reference timestamp
 * @param {AbortSignal} [signal]
 */
export async function getBuildingState(buildingId, meter = 0, refTime = null, signal) {
  const params = new URLSearchParams()
  params.set('meter', meter.toString())
  if (refTime) params.set('ref_time', refTime)
  return request(`/buildings/${encodeURIComponent(buildingId)}/state?${params.toString()}`, {
    method: 'GET',
    signal,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 24-Hour Energy Forecast
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve recursive 24-hour energy forecast.
 * @param {number} buildingId
 * @param {number} [meter=0]
 * @param {string} [cutoffTime] - Optional historical cutoff timestamp (ISO)
 * @param {AbortSignal} [signal]
 */
export async function getBuildingForecast(buildingId, meter = 0, cutoffTime = null, signal) {
  const params = new URLSearchParams()
  params.set('meter', meter.toString())
  if (cutoffTime) params.set('cutoff_time', cutoffTime)
  return request(`/buildings/${encodeURIComponent(buildingId)}/forecast?${params.toString()}`, {
    method: 'GET',
    signal,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Energy Anomalies Analysis
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve residual anomaly analysis and auxiliary Isolation Forest signal.
 * @param {number} buildingId
 * @param {number} [meter=0]
 * @param {number} [limit=168]
 * @param {boolean} [anomaliesOnly=false]
 * @param {string} [startTime]
 * @param {string} [endTime]
 * @param {AbortSignal} [signal]
 */
export async function getBuildingAnomalies(
  buildingId,
  meter = 0,
  limit = 168,
  anomaliesOnly = false,
  startTime = null,
  endTime = null,
  signal
) {
  const params = new URLSearchParams()
  params.set('meter', meter.toString())
  params.set('limit', limit.toString())
  if (anomaliesOnly) params.set('anomalies_only', 'true')
  if (startTime) params.set('start_time', startTime)
  if (endTime) params.set('end_time', endTime)
  return request(`/buildings/${encodeURIComponent(buildingId)}/anomalies?${params.toString()}`, {
    method: 'GET',
    signal,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Cost and Carbon Impact
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve 24-hour electricity cost (₹) and carbon emissions (kg CO2e) calculations.
 * @param {number} buildingId
 * @param {number} [meter=0]
 * @param {number} [tariff] - Custom tariff in ₹/kWh
 * @param {number} [carbonIntensity] - Custom carbon intensity in kg/kWh
 * @param {string} [cutoffTime]
 * @param {AbortSignal} [signal]
 */
export async function getBuildingImpact(
  buildingId,
  meter = 0,
  tariff = null,
  carbonIntensity = null,
  cutoffTime = null,
  signal
) {
  const params = new URLSearchParams()
  params.set('meter', meter.toString())
  if (tariff !== null && tariff !== undefined) params.set('tariff_per_kwh', tariff.toString())
  if (carbonIntensity !== null && carbonIntensity !== undefined) {
    params.set('carbon_intensity_kg_per_kwh', carbonIntensity.toString())
  }
  if (cutoffTime) params.set('cutoff_time', cutoffTime)
  return request(`/buildings/${encodeURIComponent(buildingId)}/impact?${params.toString()}`, {
    method: 'GET',
    signal,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Energy Recommendations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve deterministic priority-ordered recommendations from TwinState.
 * @param {number} buildingId
 * @param {number} [meter=0]
 * @param {number} [costThreshold]
 * @param {number} [carbonThreshold]
 * @param {string} [cutoffTime]
 * @param {AbortSignal} [signal]
 */
export async function getBuildingRecommendations(
  buildingId,
  meter = 0,
  costThreshold = null,
  carbonThreshold = null,
  cutoffTime = null,
  signal
) {
  const params = new URLSearchParams()
  params.set('meter', meter.toString())
  if (costThreshold !== null && costThreshold !== undefined) {
    params.set('cost_threshold', costThreshold.toString())
  }
  if (carbonThreshold !== null && carbonThreshold !== undefined) {
    params.set('carbon_threshold', carbonThreshold.toString())
  }
  if (cutoffTime) params.set('cutoff_time', cutoffTime)
  return request(`/buildings/${encodeURIComponent(buildingId)}/recommendations?${params.toString()}`, {
    method: 'GET',
    signal,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. What-If Simulation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run what-if energy demand simulation on forecast baseline.
 * @param {number} buildingId
 * @param {{
 *   scenario_type: 'ENERGY_MULTIPLIER' | 'ENERGY_REDUCTION' | 'FIXED_KWH_ADDITION',
 *   value: number,
 *   meter?: number,
 *   tariff_per_kwh?: number,
 *   carbon_intensity_kg_per_kwh?: number,
 *   cutoff_time?: string
 * }} payload
 * @param {AbortSignal} [signal]
 */
export async function simulateBuildingDemand(buildingId, payload, signal) {
  return request(`/buildings/${encodeURIComponent(buildingId)}/simulate`, {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  })
}
