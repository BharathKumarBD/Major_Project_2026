import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  TrendingUp,
  AlertOctagon,
  IndianRupee,
  Cpu,
  Lightbulb,
  Sparkles,
  RefreshCw,
} from 'lucide-react'

import Header from './components/Header'
import EnergyOverview from './components/EnergyOverview'
import ForecastPanel from './components/ForecastPanel'
import AnomalyPanel from './components/AnomalyPanel'
import ImpactPanel from './components/ImpactPanel'
import TwinStatePanel from './components/TwinStatePanel'
import RecommendationsPanel from './components/RecommendationsPanel'
import SimulationPanel from './components/SimulationPanel'
import LoadingSpinner from './components/common/LoadingSpinner'
import ErrorMessage from './components/common/ErrorMessage'

import {
  getHealth,
  getBuildings,
  getBuildingState,
  getBuildingForecast,
  getBuildingAnomalies,
  getBuildingImpact,
  getBuildingRecommendations,
  simulateBuildingDemand,
} from './services/api'

export default function App() {
  // Global API & selection state
  const [apiStatus, setApiStatus] = useState('checking') // 'connected' | 'disconnected'
  const [buildings, setBuildings] = useState([])
  const [meters, setMeters] = useState([
    { meter_id: 0, name: 'Electricity' },
    { meter_id: 1, name: 'Chilled Water' },
    { meter_id: 2, name: 'Steam' },
    { meter_id: 3, name: 'Hot Water' },
  ])
  const [selectedBuilding, setSelectedBuilding] = useState(0)
  const [selectedMeter, setSelectedMeter] = useState(0)

  // Active dashboard tab
  const [activeTab, setActiveTab] = useState('forecast')

  // Feature data states
  const [stateData, setStateData] = useState(null)
  const [forecastData, setForecastData] = useState(null)
  const [anomalyData, setAnomalyData] = useState(null)
  const [impactData, setImpactData] = useState(null)
  const [recommendationsData, setRecommendationsData] = useState(null)
  const [simulationResult, setSimulationResult] = useState(null)

  // Loading states
  const [isLoadingBuildings, setIsLoadingBuildings] = useState(true)
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [isSimulating, setIsSimulating] = useState(false)
  const [isUpdatingImpact, setIsUpdatingImpact] = useState(false)

  // Error states
  const [globalError, setGlobalError] = useState(null)
  const [simulationError, setSimulationError] = useState(null)

  // Ref to active AbortController for cancelling stale requests
  const activeAbortControllerRef = useRef(null)

  // 1. Initial Health and Buildings Load
  useEffect(() => {
    let isMounted = true

    async function initialize() {
      try {
        await getHealth()
        if (isMounted) setApiStatus('connected')
      } catch (err) {
        if (isMounted) setApiStatus('disconnected')
      }

      try {
        setIsLoadingBuildings(true)
        const bRes = await getBuildings()
        if (isMounted) {
          if (bRes.building_ids && bRes.building_ids.length > 0) {
            setBuildings(bRes.building_ids)
            // Keep default building 0 if available, else first in list
            if (!bRes.building_ids.includes(selectedBuilding)) {
              setSelectedBuilding(bRes.building_ids[0])
            }
          }
          if (bRes.meters && bRes.meters.length > 0) {
            setMeters(bRes.meters)
          }
        }
      } catch (err) {
        if (isMounted) {
          setGlobalError(err.message || 'Failed to retrieve buildings list from API.')
        }
      } finally {
        if (isMounted) setIsLoadingBuildings(false)
      }
    }

    initialize()

    return () => {
      isMounted = false
    }
  }, [])

  // 2. Fetch Building Data when building or meter changes
  const fetchBuildingData = useCallback(async (buildingId, meter) => {
    // Cancel previous in-flight requests
    if (activeAbortControllerRef.current) {
      activeAbortControllerRef.current.abort()
    }

    const controller = new AbortController()
    activeAbortControllerRef.current = controller
    const signal = controller.signal

    setIsLoadingData(true)
    setGlobalError(null)
    setSimulationResult(null)
    setSimulationError(null)

    try {
      // Execute all 5 component queries in parallel with individual resilience
      const [stateRes, forecastRes, anomRes, impactRes, recsRes] = await Promise.allSettled([
        getBuildingState(buildingId, meter, null, signal),
        getBuildingForecast(buildingId, meter, null, signal),
        getBuildingAnomalies(buildingId, meter, 168, false, null, null, signal),
        getBuildingImpact(buildingId, meter, null, null, null, signal),
        getBuildingRecommendations(buildingId, meter, null, null, null, signal),
      ])

      // Only update state if this request wasn't aborted
      if (!signal.aborted) {
        // State
        if (stateRes.status === 'fulfilled') {
          setStateData(stateRes.value)
        } else {
          setStateData(null)
        }

        // Forecast
        if (forecastRes.status === 'fulfilled') {
          setForecastData(forecastRes.value)
        } else {
          setForecastData(null)
        }

        // Anomalies
        if (anomRes.status === 'fulfilled') {
          setAnomalyData(anomRes.value)
        } else {
          setAnomalyData(null)
        }

        // Impact
        if (impactRes.status === 'fulfilled') {
          setImpactData(impactRes.value)
        } else {
          setImpactData(null)
        }

        // Recommendations
        if (recsRes.status === 'fulfilled') {
          setRecommendationsData(recsRes.value)
        } else {
          setRecommendationsData(null)
        }

        // Check if all endpoints rejected
        const allFailed = [stateRes, forecastRes, anomRes, impactRes, recsRes].every(
          (r) => r.status === 'rejected'
        )
        if (allFailed) {
          const firstErr = [stateRes, forecastRes, anomRes, impactRes, recsRes][0].reason
          setGlobalError(firstErr?.message || 'Failed to load building data from API.')
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setGlobalError(err.message || 'Error occurred while querying building data.')
      }
    } finally {
      if (!signal.aborted) {
        setIsLoadingData(false)
      }
    }
  }, [])

  // Trigger data load on building/meter change
  useEffect(() => {
    fetchBuildingData(selectedBuilding, selectedMeter)
  }, [selectedBuilding, selectedMeter, fetchBuildingData])

  // 3. What-If Demand Simulation Runner
  const handleRunSimulation = async (payload) => {
    setIsSimulating(true)
    setSimulationError(null)

    try {
      const res = await simulateBuildingDemand(selectedBuilding, payload)
      setSimulationResult(res)
    } catch (err) {
      setSimulationError(err.message || 'Simulation execution failed.')
    } finally {
      setIsSimulating(false)
    }
  }

  // 4. Update Cost/Carbon Impact Assumptions
  const handleUpdateAssumptions = async (tariff, carbon) => {
    setIsUpdatingImpact(true)
    try {
      const res = await getBuildingImpact(selectedBuilding, selectedMeter, tariff, carbon)
      setImpactData(res)
    } catch (err) {
      setGlobalError(`Failed to update assumptions: ${err.message}`)
    } finally {
      setIsUpdatingImpact(false)
    }
  }

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        apiStatus={apiStatus}
        buildings={buildings}
        meters={meters}
        selectedBuilding={selectedBuilding}
        selectedMeter={selectedMeter}
        onSelectBuilding={setSelectedBuilding}
        onSelectMeter={setSelectedMeter}
        isLoadingBuildings={isLoadingBuildings}
      />

      {/* Main Dashboard Body */}
      <main className="main-content">
        {/* Global Error Alert */}
        {globalError && (
          <ErrorMessage
            title="API Data Fetch Notice"
            message={globalError}
            onRetry={() => fetchBuildingData(selectedBuilding, selectedMeter)}
          />
        )}

        {/* Top KPI Energy Overview */}
        <EnergyOverview
          stateData={stateData}
          forecastData={forecastData}
          impactData={impactData}
        />

        {/* Tab Navigation */}
        <nav className="tab-nav" aria-label="Dashboard views">
          <button
            onClick={() => setActiveTab('forecast')}
            className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
            aria-selected={activeTab === 'forecast'}
            role="tab"
          >
            <TrendingUp size={16} /> 24h Forecast
          </button>
          <button
            onClick={() => setActiveTab('anomalies')}
            className={`tab-btn ${activeTab === 'anomalies' ? 'active' : ''}`}
            aria-selected={activeTab === 'anomalies'}
            role="tab"
          >
            <AlertOctagon size={16} /> Anomaly Detection
          </button>
          <button
            onClick={() => setActiveTab('impact')}
            className={`tab-btn ${activeTab === 'impact' ? 'active' : ''}`}
            aria-selected={activeTab === 'impact'}
            role="tab"
          >
            <IndianRupee size={16} /> Cost & Carbon Impact
          </button>
          <button
            onClick={() => setActiveTab('twin_state')}
            className={`tab-btn ${activeTab === 'twin_state' ? 'active' : ''}`}
            aria-selected={activeTab === 'twin_state'}
            role="tab"
          >
            <Cpu size={16} /> Twin State Inspector
          </button>
          <button
            onClick={() => setActiveTab('recommendations')}
            className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`}
            aria-selected={activeTab === 'recommendations'}
            role="tab"
          >
            <Lightbulb size={16} /> Recommendations
          </button>
          <button
            onClick={() => setActiveTab('simulation')}
            className={`tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
            aria-selected={activeTab === 'simulation'}
            role="tab"
          >
            <Sparkles size={16} /> What-If Simulation
          </button>
        </nav>

        {/* Active Tab View */}
        {isLoadingData ? (
          <div className="glass-card">
            <LoadingSpinner text={`Querying Twinergy API for Building #${selectedBuilding}...`} />
          </div>
        ) : (
          <div>
            {activeTab === 'forecast' && (
              <ForecastPanel forecastData={forecastData} />
            )}

            {activeTab === 'anomalies' && (
              <AnomalyPanel anomalyData={anomalyData} />
            )}

            {activeTab === 'impact' && (
              <ImpactPanel
                impactData={impactData}
                onUpdateAssumptions={handleUpdateAssumptions}
                isUpdating={isUpdatingImpact}
              />
            )}

            {activeTab === 'twin_state' && (
              <TwinStatePanel stateData={stateData} />
            )}

            {activeTab === 'recommendations' && (
              <RecommendationsPanel recommendationsData={recommendationsData} />
            )}

            {activeTab === 'simulation' && (
              <SimulationPanel
                selectedBuilding={selectedBuilding}
                selectedMeter={selectedMeter}
                onRunSimulation={handleRunSimulation}
                simulationResult={simulationResult}
                isLoading={isSimulating}
                error={simulationError}
              />
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '20px 24px',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-dim)',
          background: 'rgba(6, 9, 19, 0.9)',
        }}
      >
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <strong>Twinergy 2.0</strong> · AI-Driven Digital Twin for Building Energy Intelligence
          </div>
          <div style={{ display: 'flex', gap: 16 }}>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              style={{ color: 'var(--cyan-primary)', textDecoration: 'none' }}
            >
              Swagger API Docs ↗
            </a>
            <a
              href="http://127.0.0.1:8000/redoc"
              target="_blank"
              rel="noreferrer"
              style={{ color: 'var(--text-muted)', textDecoration: 'none' }}
            >
              ReDoc ↗
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}
