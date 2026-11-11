"""Pydantic data schemas for Twinergy REST API.

Provides structured request and response models with strict validation,
type safety, and clear schema documentation for OpenAPI/Swagger UI.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health Schema
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health status model."""
    status: str = Field("healthy", description="Current service health status")
    service: str = Field("Twinergy API", description="Name of the API service")
    version: str = Field("2.0.0", description="API version")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "service": "Twinergy API",
            "version": "2.0.0"
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. Buildings Schema
# ─────────────────────────────────────────────────────────────────────────────

class MeterInfo(BaseModel):
    """Meter type identifier and descriptive name."""
    meter_id: int = Field(..., description="Integer meter identifier (e.g. 0 for Electricity)")
    name: str = Field(..., description="Canonical meter utility name")


class BuildingsResponse(BaseModel):
    """List of available buildings and supported meter types."""
    total_buildings: int = Field(..., description="Total count of unique buildings returned")
    building_ids: List[int] = Field(..., description="Sorted list of building IDs")
    meters: List[MeterInfo] = Field(..., description="List of distinct meter types available")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_buildings": 2,
            "building_ids": [0, 1],
            "meters": [
                {"meter_id": 0, "name": "Electricity"},
                {"meter_id": 1, "name": "Chilled Water"},
                {"meter_id": 2, "name": "Steam"},
                {"meter_id": 3, "name": "Hot Water"}
            ]
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3. Twin State Schema
# ─────────────────────────────────────────────────────────────────────────────

class TwinStateResponse(BaseModel):
    """Software representation of building energy state and derived intelligence."""
    building_id: int = Field(..., description="ASHRAE building identifier")
    timestamp: str = Field(..., description="State reference timestamp (ISO format)")
    meter: int = Field(0, description="Meter identifier (0 for Electricity)")

    # Current Energy
    actual_kwh: Optional[float] = Field(None, description="Observed energy consumption in kWh")
    expected_kwh: Optional[float] = Field(None, description="Expected baseline consumption in kWh")
    residual_kwh: Optional[float] = Field(None, description="Deviation (actual - expected) in kWh")
    deviation_ratio: Optional[float] = Field(None, description="Normalized deviation ratio")

    # Anomaly
    severity: Optional[str] = Field(None, description="Anomaly severity: NORMAL, WARNING, CRITICAL")
    anomaly_type: Optional[str] = Field(None, description="Deterministic anomaly category")
    classification_reason: Optional[str] = Field(None, description="Explainable classification reason")

    # Forecast
    forecast_horizon_hours: Optional[int] = Field(None, description="Forecast horizon in hours (24)")
    forecast_total_kwh: Optional[float] = Field(None, description="Cumulative 24h forecasted energy (kWh)")
    forecast_peak_kwh: Optional[float] = Field(None, description="Peak predicted hourly consumption (kWh)")
    forecast_peak_hour: Optional[str] = Field(None, description="Timestamp of peak predicted consumption")

    # Impact
    forecast_total_cost: Optional[float] = Field(None, description="Total 24h forecasted cost (₹)")
    forecast_total_carbon_kg: Optional[float] = Field(None, description="Total 24h forecasted carbon (kg CO2e)")

    # Metadata & Status
    weather_source: Optional[str] = Field(None, description="Source of weather observations")
    state_status: str = Field("UNAVAILABLE", description="Aggregated state status: NORMAL, WARNING, CRITICAL, UNAVAILABLE")
    generated_at: str = Field(..., description="ISO timestamp when this state was instantiated")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forecast Schema
# ─────────────────────────────────────────────────────────────────────────────

class ForecastHourlyItem(BaseModel):
    """Hourly forecast prediction and environmental context."""
    step: int = Field(..., ge=1, le=24, description="Horizon step (1 to 24)")
    timestamp: str = Field(..., description="Forecast timestamp (ISO format)")
    predicted_kwh: float = Field(..., ge=0.0, description="Predicted consumption in physical kWh")
    predicted_log: float = Field(..., description="Predicted consumption on log1p scale")
    air_temperature: Optional[float] = Field(None, description="Air temperature at horizon hour (°C)")


class ForecastSummary(BaseModel):
    """Aggregate metrics over the 24-hour forecast horizon."""
    total_kwh: float = Field(..., ge=0.0, description="Cumulative 24-hour predicted energy (kWh)")
    peak_kwh: float = Field(..., ge=0.0, description="Maximum predicted hourly energy (kWh)")
    peak_timestamp: str = Field(..., description="Timestamp of peak predicted demand")
    min_kwh: float = Field(..., ge=0.0, description="Minimum predicted hourly energy (kWh)")
    min_timestamp: str = Field(..., description="Timestamp of minimum predicted demand")
    average_kwh: float = Field(..., ge=0.0, description="Mean hourly predicted energy (kWh)")


class ForecastResponse(BaseModel):
    """Recursive 24-hour energy forecast response."""
    building_id: int = Field(..., description="Building identifier")
    meter: int = Field(0, description="Meter identifier")
    cutoff_time: str = Field(..., description="Historical cutoff timestamp (ISO format)")
    weather_source: str = Field(..., description="'dataset_observed' or 'provided'")
    summary: ForecastSummary = Field(..., description="Forecast horizon summary metrics")
    hourly_forecast: List[ForecastHourlyItem] = Field(..., description="List of 24 hourly predictions")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Anomalies Schema
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyHourlyItem(BaseModel):
    """Hourly consumption record with residual evaluation and auxiliary signal."""
    timestamp: str = Field(..., description="Observation timestamp")
    actual_kwh: float = Field(..., description="Observed energy consumption in kWh")
    expected_kwh: float = Field(..., description="Model expected energy baseline in kWh")
    residual: float = Field(..., description="Residual (actual - expected) in kWh")
    deviation_ratio: float = Field(..., description="Relative deviation ratio (residual / expected)")
    severity: str = Field(..., description="NORMAL, WARNING, or CRITICAL")
    is_anomaly: int = Field(..., description="1 if anomaly, 0 if normal")
    anomaly_direction: str = Field(..., description="HIGH_DEMAND, LOW_DEMAND, or NOMINAL")
    anomaly_type: str = Field(..., description="NORMAL, HIGH_DEMAND, LOW_DEMAND, RAPID_SPIKE, PERSISTENT_HIGH")
    classification_reason: str = Field(..., description="Deterministic explainable reason")
    isoforest_anomaly: Optional[int] = Field(None, description="Auxiliary Isolation Forest label (1=anomaly, 0=normal)")
    isoforest_score: Optional[float] = Field(None, description="Auxiliary Isolation Forest score")


class PrimaryResidualSignalSummary(BaseModel):
    """Summary of primary residual-based anomaly detection."""
    warning_threshold: float = Field(0.25, description="Deviation ratio threshold for WARNING (25%)")
    critical_threshold: float = Field(0.50, description="Deviation ratio threshold for CRITICAL (50%)")
    counts_by_severity: Dict[str, int] = Field(..., description="Count of observations per severity level")
    counts_by_type: Dict[str, int] = Field(..., description="Count of observations per deterministic anomaly type")
    causality_warning: str = Field(..., description="Scientific non-causality advisory")


class AuxiliaryIsolationForestSummary(BaseModel):
    """Summary of auxiliary Isolation Forest detector."""
    anomaly_count: int = Field(..., description="Count of observations flagged by Isolation Forest")
    training_scope: str = Field("Building 20 only (Auxiliary signal)", description="Training limitation note")
    warning: str = Field(..., description="Model artifact scope warning")


class AnomaliesResponse(BaseModel):
    """Building anomaly evaluation response with separated signals."""
    building_id: int = Field(..., description="Building identifier")
    meter: int = Field(0, description="Meter identifier")
    window_start: str = Field(..., description="Start timestamp of evaluated window")
    window_end: str = Field(..., description="End timestamp of evaluated window")
    total_records: int = Field(..., description="Total hourly records evaluated")
    anomaly_count: int = Field(..., description="Total observations flagged by primary residual engine")
    primary_residual_signal: PrimaryResidualSignalSummary = Field(..., description="Primary residual analysis")
    auxiliary_isolation_forest_signal: AuxiliaryIsolationForestSummary = Field(..., description="Auxiliary Isolation Forest signal")
    anomalies: List[AnomalyHourlyItem] = Field(..., description="List of evaluated hourly records")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cost & Carbon Impact Schema
# ─────────────────────────────────────────────────────────────────────────────

class ImpactAssumptions(BaseModel):
    """Configurable tariff and emissions factors."""
    tariff_per_kwh: float = Field(8.0, ge=0.0, description="Electricity tariff rate (₹/kWh)")
    carbon_intensity_kg_per_kwh: float = Field(0.70, ge=0.0, description="Grid carbon intensity (kg CO2e/kWh)")
    disclaimer: str = Field(
        "Demonstration assumptions; not actual utility tariff or measured grid carbon intensity.",
        description="Assumptions disclaimer"
    )


class ImpactHourlyItem(BaseModel):
    """Hourly forecast impact metric."""
    step: int = Field(..., ge=1, le=24, description="Forecast step index (1 to 24)")
    timestamp: str = Field(..., description="Forecast hourly timestamp (ISO format)")
    predicted_kwh: float = Field(..., ge=0.0, description="Forecasted energy demand (kWh)")
    hourly_cost: float = Field(..., ge=0.0, description="Calculated hourly energy cost (₹)")
    hourly_carbon_kg: float = Field(..., ge=0.0, description="Calculated hourly carbon emissions (kg CO2e)")


class ImpactSummary(BaseModel):
    """Cumulative 24h impact forecast totals and averages."""
    total_kwh: float = Field(..., ge=0.0, description="Total 24h predicted energy (kWh)")
    total_cost: float = Field(..., ge=0.0, description="Total 24h predicted cost (₹)")
    total_carbon_kg: float = Field(..., ge=0.0, description="Total 24h predicted carbon (kg CO2e)")
    average_hourly_cost: float = Field(..., ge=0.0, description="Average hourly cost (₹)")
    average_hourly_carbon_kg: float = Field(..., ge=0.0, description="Average hourly carbon emissions (kg CO2e)")
    peak_cost_hour: Optional[str] = Field(None, description="Timestamp of peak hourly cost")
    peak_carbon_hour: Optional[str] = Field(None, description="Timestamp of peak hourly carbon")


class ImpactResponse(BaseModel):
    """24-hour cost and carbon impact response."""
    building_id: int = Field(..., description="Building identifier")
    meter: int = Field(0, description="Meter identifier")
    cutoff_time: str = Field(..., description="Historical cutoff timestamp used for forecast")
    assumptions: ImpactAssumptions = Field(..., description="Tariff and carbon factor assumptions")
    summary: ImpactSummary = Field(..., description="Impact totals and averages")
    hourly_impact: List[ImpactHourlyItem] = Field(..., description="Hourly impact trajectory")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Recommendations Schema
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    """Structured energy management recommendation."""
    recommendation_id: str = Field(..., description="Unique recommendation ID (e.g. REC-COST-001)")
    priority: str = Field(..., description="Priority: CRITICAL, HIGH, MEDIUM, LOW, INFO")
    category: str = Field(..., description="Category: ENERGY, ANOMALY, COST, CARBON, OPERATIONS")
    title: str = Field(..., description="Concise recommendation title")
    action: str = Field(..., description="Non-causal actionable advisory statement")
    reason: str = Field(..., description="Trigger justification reason")
    evidence: str = Field(..., description="Quantitative evidence from twin state / forecast")


class RecommendationsResponse(BaseModel):
    """Ordered advisory recommendations response."""
    building_id: int = Field(..., description="Building identifier")
    meter: int = Field(0, description="Meter identifier")
    twin_state_status: str = Field(..., description="Building Twin State status at time of recommendation")
    total_count: int = Field(..., description="Total count of recommendations generated")
    generated_at: str = Field(..., description="Timestamp of recommendation generation")
    recommendations: List[RecommendationItem] = Field(..., description="Priority-ordered recommendations")


# ─────────────────────────────────────────────────────────────────────────────
# 8. What-If Simulation Schema
# ─────────────────────────────────────────────────────────────────────────────

ValidScenarioType = Literal["ENERGY_MULTIPLIER", "ENERGY_REDUCTION", "FIXED_KWH_ADDITION"]


class SimulationRequest(BaseModel):
    """Request model for what-if energy demand simulation."""
    scenario_type: ValidScenarioType = Field(
        ...,
        description="Scenario type: ENERGY_MULTIPLIER, ENERGY_REDUCTION, or FIXED_KWH_ADDITION"
    )
    value: float = Field(
        ...,
        description="Scenario parameter (multiplier factor >0, reduction fraction [0,1), or hourly kWh addition)"
    )
    meter: int = Field(0, ge=0, description="Meter identifier (default 0 for Electricity)")
    tariff_per_kwh: Optional[float] = Field(None, ge=0.0, description="Optional tariff override (₹/kWh)")
    carbon_intensity_kg_per_kwh: Optional[float] = Field(None, ge=0.0, description="Optional carbon intensity override (kg/kWh)")
    cutoff_time: Optional[str] = Field(None, description="Optional ISO cutoff timestamp for forecast horizon")

    @field_validator("value")
    @classmethod
    def validate_scenario_value(cls, v: float, info) -> float:
        st = info.data.get("scenario_type")
        if st == "ENERGY_MULTIPLIER" and v <= 0.0:
            raise ValueError(f"ENERGY_MULTIPLIER value must be > 0, got {v}")
        if st == "ENERGY_REDUCTION" and (v < 0.0 or v >= 1.0):
            raise ValueError(f"ENERGY_REDUCTION fraction must be in range [0, 1), got {v}")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "scenario_type": "ENERGY_MULTIPLIER",
            "value": 1.10,
            "meter": 0,
            "tariff_per_kwh": 8.0,
            "carbon_intensity_kg_per_kwh": 0.70
        }
    })


class SimulationHourlyItem(BaseModel):
    """Hourly simulated energy trajectory comparison."""
    timestamp: str = Field(..., description="Hourly timestamp (ISO format)")
    baseline_kwh: float = Field(..., ge=0.0, description="Baseline forecasted demand (kWh)")
    simulated_kwh: float = Field(..., ge=0.0, description="Simulated demand under scenario (kWh)")
    delta_kwh: float = Field(..., description="Absolute change (simulated - baseline) in kWh")
    delta_percent: float = Field(..., description="Percentage change in consumption (%)")


class SimulationSummary(BaseModel):
    """Aggregate simulation metrics comparing baseline and simulated demand."""
    baseline_total_kwh: float = Field(..., description="Baseline 24h cumulative energy (kWh)")
    simulated_total_kwh: float = Field(..., description="Simulated 24h cumulative energy (kWh)")
    delta_total_kwh: float = Field(..., description="Total energy delta (kWh)")
    delta_total_percent: float = Field(..., description="Total energy change (%)")

    baseline_peak_kwh: float = Field(..., description="Baseline peak hourly consumption (kWh)")
    simulated_peak_kwh: float = Field(..., description="Simulated peak hourly consumption (kWh)")
    baseline_peak_hour: Optional[str] = Field(None, description="Timestamp of baseline peak demand")
    simulated_peak_hour: Optional[str] = Field(None, description="Timestamp of simulated peak demand")

    baseline_total_cost: float = Field(..., description="Baseline 24h energy cost (₹)")
    simulated_total_cost: float = Field(..., description="Simulated 24h energy cost (₹)")
    delta_total_cost: float = Field(..., description="Cost delta (₹)")

    baseline_total_carbon_kg: float = Field(..., description="Baseline 24h carbon emissions (kg CO2e)")
    simulated_total_carbon_kg: float = Field(..., description="Simulated 24h carbon emissions (kg CO2e)")
    delta_total_carbon_kg: float = Field(..., description="Carbon emissions delta (kg CO2e)")

    tariff_per_kwh: float = Field(..., description="Tariff rate used in calculation (₹/kWh)")
    carbon_intensity_kg_per_kwh: float = Field(..., description="Carbon intensity factor used (kg/kWh)")


class SimulationResponse(BaseModel):
    """What-if demand simulation results response."""
    building_id: int = Field(..., description="Building identifier")
    meter: int = Field(0, description="Meter identifier")
    scenario_type: str = Field(..., description="Simulated scenario type")
    scenario_value: float = Field(..., description="Scenario magnitude value")
    summary: SimulationSummary = Field(..., description="Summary impact comparison")
    hourly_simulation: List[SimulationHourlyItem] = Field(..., description="Hourly trajectory comparison")
