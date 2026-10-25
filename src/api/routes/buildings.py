"""Building digital twin routes for Twinergy 2.0 API.

Provides endpoints for:
- GET /buildings
- GET /buildings/{building_id}/state
- GET /buildings/{building_id}/forecast
- GET /buildings/{building_id}/anomalies
- GET /buildings/{building_id}/impact
- GET /buildings/{building_id}/recommendations
- POST /buildings/{building_id}/simulate
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    AnomaliesResponse,
    AnomalyHourlyItem,
    AuxiliaryIsolationForestSummary,
    BuildingsResponse,
    ForecastHourlyItem,
    ForecastResponse,
    ForecastSummary,
    ImpactAssumptions,
    ImpactHourlyItem,
    ImpactResponse,
    ImpactSummary,
    MeterInfo,
    PrimaryResidualSignalSummary,
    RecommendationItem,
    RecommendationsResponse,
    SimulationHourlyItem,
    SimulationRequest,
    SimulationResponse,
    SimulationSummary,
    TwinStateResponse,
)
from src.api.services import (
    get_anomaly_analyzer,
    get_anomaly_detector,
    get_data_loader,
    get_forecaster,
    get_impact_calculator,
    get_recommendation_engine,
    get_simulation_engine,
    get_twin_state_builder,
    sanitize_error_message,
    to_iso_string,
    validate_building_and_meter,
)
from src.data.loader import DataLoader
from src.data.schema import METER_TYPE_NAMES
from src.models.anomaly import (
    BUILDING_20_ISOFOREST_WARNING,
    CAUSALITY_WARNING,
    AnomalyDetector,
    EnergyAnomalyAnalyzer,
)
from src.models.forecasting import EnergyForecaster
from src.models.impact import EnergyImpactCalculator
from src.models.recommendations import RecommendationEngine
from src.models.simulation import EnergySimulationEngine
from src.models.twin_state import TwinStateBuilder, build_twin_state

logger = logging.getLogger("twinergy.api.buildings")

router = APIRouter(prefix="/buildings", tags=["Buildings"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /buildings
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=BuildingsResponse,
    summary="List Available Buildings and Meter Types",
    description="Returns list of unique building IDs and available meter types using DuckDB pushdown queries without loading the full CSV into memory.",
)
def list_buildings(
    meter: Optional[int] = Query(None, ge=0, description="Optional meter ID to filter building IDs"),
    loader: DataLoader = Depends(get_data_loader),
) -> BuildingsResponse:
    """Return available building IDs and meter types."""
    try:
        available_meters = loader.get_meters()
    except Exception as exc:
        logger.error("Failed to query meters: %s", exc)
        available_meters = [0, 1, 2, 3]

    meter_infos = [
        MeterInfo(
            meter_id=m_id,
            name=METER_TYPE_NAMES.get(m_id, f"Meter {m_id}"),
        )
        for m_id in available_meters
    ]

    try:
        building_ids = loader.get_building_ids(meter=meter)
    except Exception as exc:
        logger.error("Failed to query building IDs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve building list from dataset.",
        )

    return BuildingsResponse(
        total_buildings=len(building_ids),
        building_ids=building_ids,
        meters=meter_infos,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /buildings/{building_id}/state
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{building_id}/state",
    response_model=TwinStateResponse,
    summary="Get Building Digital Twin State",
    description="Constructs and returns the existing TwinState representation aggregating current energy, anomaly classification, forecast, and impact state for a specific reference timestamp.",
)
def get_building_state(
    building_id: int,
    meter: int = Query(0, ge=0, description="Meter type (default 0 for Electricity)"),
    ref_time: Optional[str] = Query(None, description="Optional ISO timestamp for state evaluation"),
    loader: DataLoader = Depends(get_data_loader),
    forecaster: EnergyForecaster = Depends(get_forecaster),
    detector: AnomalyDetector = Depends(get_anomaly_detector),
    impact_calc: EnergyImpactCalculator = Depends(get_impact_calculator),
    twin_builder: TwinStateBuilder = Depends(get_twin_state_builder),
) -> TwinStateResponse:
    """Return software TwinState representation for building and meter."""
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    # 1. Resolve reference timestamp
    if ref_time is not None:
        try:
            ref_ts = pd.to_datetime(ref_time)
            if pd.isna(ref_ts):
                raise ValueError("Invalid timestamp")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid ref_time format: '{ref_time}'. Please provide a valid ISO timestamp.",
            )
    else:
        ref_ts = latest_ts

    # 2. Attempt 24-hour forecast from cutoff = ref_ts - 24 hours
    cutoff = ref_ts - pd.Timedelta(hours=24)
    forecast_df = None
    impact_dict = None

    try:
        forecast_df = forecaster.forecast_24h(
            building_id=building_id,
            meter=meter,
            cutoff_time=cutoff,
            loader=loader,
        )
        impact_dict = impact_calc.calculate_impact(forecast_df)
    except Exception as exc:
        logger.warning(
            "Forecast computation skipped for TwinState (building=%s, meter=%s): %s",
            building_id,
            meter,
            exc,
        )
        forecast_df = None
        impact_dict = None

    # 3. Attempt anomaly detection on recent historical window up to ref_ts
    anom_df = None
    try:
        recent_start = ref_ts - pd.Timedelta(hours=24)
        recent_df = loader.load(
            building_id=building_id,
            meter=meter,
            start_time=recent_start,
            end_time=ref_ts,
        )
        if len(recent_df) > 0:
            anom_df = detector.detect(recent_df)
    except Exception as exc:
        logger.warning(
            "Anomaly evaluation skipped for TwinState (building=%s, meter=%s): %s",
            building_id,
            meter,
            exc,
        )
        anom_df = None

    # 4. Construct validated TwinState object
    try:
        twin_state = twin_builder.build(
            building_id=building_id,
            timestamp=ref_ts,
            meter=meter,
            anomaly_analysis=anom_df,
            forecast_df=forecast_df,
            impact_dict=impact_dict,
        )
        return TwinStateResponse(**twin_state.to_dict())
    except Exception as exc:
        logger.error("Failed to build TwinState for building %s: %s", building_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to aggregate building twin state.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /buildings/{building_id}/forecast
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{building_id}/forecast",
    response_model=ForecastResponse,
    summary="Get 24-Hour Energy Forecast",
    description="Returns the existing recursive 24-hour energy forecast with hourly predicted kWh, log scale predictions, and summary statistics.",
)
def get_building_forecast(
    building_id: int,
    meter: int = Query(0, ge=0, description="Meter type (default 0 for Electricity)"),
    cutoff_time: Optional[str] = Query(None, description="Optional cutoff timestamp (forecast runs t+1 to t+24)"),
    loader: DataLoader = Depends(get_data_loader),
    forecaster: EnergyForecaster = Depends(get_forecaster),
) -> ForecastResponse:
    """Return recursive 24-hour multi-step forecast."""
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    # Resolve cutoff timestamp
    if cutoff_time is not None:
        try:
            cutoff = pd.to_datetime(cutoff_time)
            if pd.isna(cutoff):
                raise ValueError("Invalid timestamp")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid cutoff_time format: '{cutoff_time}'. Please provide a valid ISO timestamp.",
            )
    else:
        cutoff = latest_ts - pd.Timedelta(hours=24)

    try:
        fc_df = forecaster.forecast_24h(
            building_id=building_id,
            meter=meter,
            cutoff_time=cutoff,
            loader=loader,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Forecast unavailable: {sanitize_error_message(str(val_err))}",
        )
    except Exception as exc:
        logger.error("Unexpected error in forecast_24h for building %s: %s", building_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during forecast generation.",
        )

    # Compute summary metrics
    total_kwh = float(fc_df["predicted_kwh"].sum())
    peak_kwh = float(fc_df["predicted_kwh"].max())
    peak_idx = int(fc_df["predicted_kwh"].idxmax())
    peak_ts = to_iso_string(fc_df["timestamp"].iloc[peak_idx])

    min_kwh = float(fc_df["predicted_kwh"].min())
    min_idx = int(fc_df["predicted_kwh"].idxmin())
    min_ts = to_iso_string(fc_df["timestamp"].iloc[min_idx])
    avg_kwh = float(fc_df["predicted_kwh"].mean())

    weather_src = str(fc_df["weather_source"].iloc[0]) if "weather_source" in fc_df.columns else "dataset_observed"

    hourly_items: List[ForecastHourlyItem] = []
    for _, row in fc_df.iterrows():
        air_temp = float(row["air_temperature"]) if ("air_temperature" in row and not pd.isna(row["air_temperature"])) else None
        hourly_items.append(
            ForecastHourlyItem(
                step=int(row["step"]),
                timestamp=to_iso_string(row["timestamp"]) or "",
                predicted_kwh=float(row["predicted_kwh"]),
                predicted_log=float(row["predicted_log"]),
                air_temperature=air_temp,
            )
        )

    summary = ForecastSummary(
        total_kwh=total_kwh,
        peak_kwh=peak_kwh,
        peak_timestamp=peak_ts or "",
        min_kwh=min_kwh,
        min_timestamp=min_ts or "",
        average_kwh=avg_kwh,
    )

    return ForecastResponse(
        building_id=building_id,
        meter=meter,
        cutoff_time=to_iso_string(cutoff) or str(cutoff),
        weather_source=weather_src,
        summary=summary,
        hourly_forecast=hourly_items,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /buildings/{building_id}/anomalies
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{building_id}/anomalies",
    response_model=AnomaliesResponse,
    summary="Get Energy Consumption Anomalies",
    description="Evaluates energy consumption records using primary residual analysis with explainable classification, and provides auxiliary Isolation Forest anomaly signals.",
)
def get_building_anomalies(
    building_id: int,
    meter: int = Query(0, ge=0, description="Meter type (default 0 for Electricity)"),
    start_time: Optional[str] = Query(None, description="Start timestamp of evaluation window (ISO format)"),
    end_time: Optional[str] = Query(None, description="End timestamp of evaluation window (ISO format)"),
    limit: int = Query(168, ge=1, le=1000, description="Maximum hourly records to retrieve (default 168 hours = 1 week)"),
    anomalies_only: bool = Query(False, description="If True, filters returned records to only flagged anomalies (is_anomaly=1)"),
    loader: DataLoader = Depends(get_data_loader),
    detector: AnomalyDetector = Depends(get_anomaly_detector),
) -> AnomaliesResponse:
    """Return primary residual anomalies and auxiliary Isolation Forest results."""
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    # Resolve time range
    if end_time is not None:
        try:
            end_ts = pd.to_datetime(end_time)
            if pd.isna(end_ts):
                raise ValueError("Invalid end_time")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid end_time format: '{end_time}'.",
            )
    else:
        end_ts = latest_ts

    if start_time is not None:
        try:
            start_ts = pd.to_datetime(start_time)
            if pd.isna(start_ts):
                raise ValueError("Invalid start_time")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid start_time format: '{start_time}'.",
            )
    else:
        start_ts = end_ts - pd.Timedelta(hours=limit - 1)

    if start_ts > end_ts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"start_time ({start_ts}) cannot be greater than end_time ({end_ts}).",
        )

    try:
        df_records = loader.load(
            building_id=building_id,
            meter=meter,
            start_time=start_ts,
            end_time=end_ts,
        )
    except Exception as exc:
        logger.error("Failed loading data for anomalies: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load historical energy records.",
        )

    if len(df_records) == 0:
        return AnomaliesResponse(
            building_id=building_id,
            meter=meter,
            window_start=to_iso_string(start_ts) or "",
            window_end=to_iso_string(end_ts) or "",
            total_records=0,
            anomaly_count=0,
            primary_residual_signal=PrimaryResidualSignalSummary(
                warning_threshold=0.25,
                critical_threshold=0.50,
                counts_by_severity={"NORMAL": 0, "WARNING": 0, "CRITICAL": 0},
                counts_by_type={},
                causality_warning=CAUSALITY_WARNING,
            ),
            auxiliary_isolation_forest_signal=AuxiliaryIsolationForestSummary(
                anomaly_count=0,
                training_scope="Building 20 only (Auxiliary signal)",
                warning=BUILDING_20_ISOFOREST_WARNING,
            ),
            anomalies=[],
        )

    # Detect anomalies
    try:
        anom_df = detector.detect(df_records)
    except Exception as exc:
        logger.error("Anomaly detection evaluation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate anomalies on historical records.",
        )

    # Aggregate summaries
    counts_by_sev = anom_df["severity"].value_counts().to_dict()
    counts_by_typ = anom_df["anomaly_type"].value_counts().to_dict()
    total_anomalies = int((anom_df["is_anomaly"] == 1).sum())

    iso_anomaly_count = (
        int(anom_df["isoforest_anomaly"].sum())
        if "isoforest_anomaly" in anom_df.columns
        else 0
    )

    primary_summary = PrimaryResidualSignalSummary(
        warning_threshold=0.25,
        critical_threshold=0.50,
        counts_by_severity={str(k): int(v) for k, v in counts_by_sev.items()},
        counts_by_type={str(k): int(v) for k, v in counts_by_typ.items()},
        causality_warning=CAUSALITY_WARNING,
    )

    aux_summary = AuxiliaryIsolationForestSummary(
        anomaly_count=iso_anomaly_count,
        training_scope="Building 20 only (Auxiliary signal)",
        warning=BUILDING_20_ISOFOREST_WARNING,
    )

    # Filter records if anomalies_only requested
    target_rows = anom_df[anom_df["is_anomaly"] == 1] if anomalies_only else anom_df

    anomaly_items: List[AnomalyHourlyItem] = []
    for _, row in target_rows.iterrows():
        iso_flag = int(row["isoforest_anomaly"]) if "isoforest_anomaly" in row and not pd.isna(row["isoforest_anomaly"]) else None
        iso_score = float(row["isoforest_score"]) if "isoforest_score" in row and not pd.isna(row["isoforest_score"]) else None

        anomaly_items.append(
            AnomalyHourlyItem(
                timestamp=to_iso_string(row["timestamp"]) or "",
                actual_kwh=float(row["actual_kwh"]),
                expected_kwh=float(row["expected_kwh"]),
                residual=float(row["residual"]),
                deviation_ratio=float(row["deviation_ratio"]),
                severity=str(row["severity"]),
                is_anomaly=int(row["is_anomaly"]),
                anomaly_direction=str(row["anomaly_direction"]),
                anomaly_type=str(row["anomaly_type"]),
                classification_reason=str(row["classification_reason"]),
                isoforest_anomaly=iso_flag,
                isoforest_score=iso_score,
            )
        )

    return AnomaliesResponse(
        building_id=building_id,
        meter=meter,
        window_start=to_iso_string(start_ts) or "",
        window_end=to_iso_string(end_ts) or "",
        total_records=len(anom_df),
        anomaly_count=total_anomalies,
        primary_residual_signal=primary_summary,
        auxiliary_isolation_forest_signal=aux_summary,
        anomalies=anomaly_items,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. GET /buildings/{building_id}/impact
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{building_id}/impact",
    response_model=ImpactResponse,
    summary="Get 24-Hour Cost and Carbon Impact",
    description="Calculates 24-hour electricity cost (₹) and carbon emissions (kg CO2e) based on the energy forecast and configurable tariff/carbon parameters.",
)
def get_building_impact(
    building_id: int,
    meter: int = Query(0, ge=0, description="Meter type (default 0 for Electricity)"),
    tariff_per_kwh: Optional[float] = Query(None, ge=0.0, description="Custom tariff in ₹/kWh (default: 8.0)"),
    carbon_intensity_kg_per_kwh: Optional[float] = Query(None, ge=0.0, description="Custom carbon intensity in kg CO2e/kWh (default: 0.70)"),
    cutoff_time: Optional[str] = Query(None, description="Optional cutoff timestamp for forecast horizon"),
    loader: DataLoader = Depends(get_data_loader),
    forecaster: EnergyForecaster = Depends(get_forecaster),
    impact_calc: EnergyImpactCalculator = Depends(get_impact_calculator),
) -> ImpactResponse:
    """Calculate cost and carbon impacts on 24-hour forecast."""
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    # Resolve cutoff
    if cutoff_time is not None:
        try:
            cutoff = pd.to_datetime(cutoff_time)
            if pd.isna(cutoff):
                raise ValueError("Invalid timestamp")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid cutoff_time format: '{cutoff_time}'.",
            )
    else:
        cutoff = latest_ts - pd.Timedelta(hours=24)

    try:
        fc_df = forecaster.forecast_24h(
            building_id=building_id,
            meter=meter,
            cutoff_time=cutoff,
            loader=loader,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Forecast unavailable for impact calculation: {sanitize_error_message(str(val_err))}",
        )
    except Exception as exc:
        logger.error("Unexpected error in forecast_24h for impact: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during forecast generation.",
        )

    try:
        impact = impact_calc.calculate_impact(
            forecast_df=fc_df,
            tariff_per_kwh=tariff_per_kwh,
            carbon_intensity_kg_per_kwh=carbon_intensity_kg_per_kwh,
        )
    except Exception as exc:
        logger.error("Impact calculation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate cost and carbon impacts.",
        )

    hourly_df = impact["hourly_df"]
    hourly_items: List[ImpactHourlyItem] = []

    for _, row in hourly_df.iterrows():
        step_val = int(row["step"]) if "step" in row else None
        hourly_items.append(
            ImpactHourlyItem(
                step=step_val or 0,
                timestamp=to_iso_string(row["timestamp"]) or "",
                predicted_kwh=float(row["predicted_kwh"]),
                hourly_cost=float(row["hourly_cost"]),
                hourly_carbon_kg=float(row["hourly_carbon_kg"]),
            )
        )

    assumptions = ImpactAssumptions(
        tariff_per_kwh=float(impact["tariff_per_kwh"]),
        carbon_intensity_kg_per_kwh=float(impact["carbon_intensity_kg_per_kwh"]),
    )

    summary = ImpactSummary(
        total_kwh=float(impact["total_kwh"]),
        total_cost=float(impact["total_cost"]),
        total_carbon_kg=float(impact["total_carbon_kg"]),
        average_hourly_cost=float(impact["average_hourly_cost"]),
        average_hourly_carbon_kg=float(impact["average_hourly_carbon_kg"]),
        peak_cost_hour=to_iso_string(impact["peak_cost_hour"]),
        peak_carbon_hour=to_iso_string(impact["peak_carbon_hour"]),
    )

    return ImpactResponse(
        building_id=building_id,
        meter=meter,
        cutoff_time=to_iso_string(cutoff) or str(cutoff),
        assumptions=assumptions,
        summary=summary,
        hourly_impact=hourly_items,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /buildings/{building_id}/recommendations
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{building_id}/recommendations",
    response_model=RecommendationsResponse,
    summary="Get Energy Management Recommendations",
    description="Generates deterministic, non-causal energy recommendations from the building's current TwinState, anomaly signals, and forecast impact in priority order.",
)
def get_building_recommendations(
    building_id: int,
    meter: int = Query(0, ge=0, description="Meter type (default 0 for Electricity)"),
    cost_threshold: Optional[float] = Query(None, ge=0.0, description="Custom 24h cost threshold in ₹"),
    carbon_threshold: Optional[float] = Query(None, ge=0.0, description="Custom 24h carbon threshold in kg CO2e"),
    cutoff_time: Optional[str] = Query(None, description="Optional reference cutoff timestamp"),
    loader: DataLoader = Depends(get_data_loader),
    forecaster: EnergyForecaster = Depends(get_forecaster),
    detector: AnomalyDetector = Depends(get_anomaly_detector),
    impact_calc: EnergyImpactCalculator = Depends(get_impact_calculator),
    twin_builder: TwinStateBuilder = Depends(get_twin_state_builder),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> RecommendationsResponse:
    """Generate deterministic recommendations from TwinState."""
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    if cutoff_time is not None:
        try:
            ref_ts = pd.to_datetime(cutoff_time)
            if pd.isna(ref_ts):
                raise ValueError("Invalid timestamp")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid cutoff_time format: '{cutoff_time}'.",
            )
    else:
        ref_ts = latest_ts

    cutoff = ref_ts - pd.Timedelta(hours=24)

    # 1. Forecast & Impact
    forecast_df = None
    impact_dict = None
    try:
        forecast_df = forecaster.forecast_24h(
            building_id=building_id,
            meter=meter,
            cutoff_time=cutoff,
            loader=loader,
        )
        impact_dict = impact_calc.calculate_impact(forecast_df)
    except Exception as exc:
        logger.warning("Forecast skipped for recommendations: %s", exc)

    # 2. Anomaly Analysis
    anom_df = None
    try:
        recent_df = loader.load(
            building_id=building_id,
            meter=meter,
            start_time=cutoff,
            end_time=ref_ts,
        )
        if len(recent_df) > 0:
            anom_df = detector.detect(recent_df)
    except Exception as exc:
        logger.warning("Anomaly analysis skipped for recommendations: %s", exc)

    # 3. Build Twin State
    twin_state = twin_builder.build(
        building_id=building_id,
        timestamp=ref_ts,
        meter=meter,
        anomaly_analysis=anom_df,
        forecast_df=forecast_df,
        impact_dict=impact_dict,
    )

    # 4. Generate recommendations
    try:
        rec_output = rec_engine.recommend(
            twin_state=twin_state,
            cost_threshold=cost_threshold,
            carbon_threshold=carbon_threshold,
        )
    except Exception as exc:
        logger.error("Recommendation generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations.",
        )

    raw_recs = rec_output.get("recommendations", [])
    rec_items = [
        RecommendationItem(
            recommendation_id=r["recommendation_id"],
            priority=r["priority"],
            category=r["category"],
            title=r["title"],
            action=r["action"],
            reason=r["reason"],
            evidence=r["evidence"],
        )
        for r in raw_recs
    ]

    return RecommendationsResponse(
        building_id=building_id,
        meter=meter,
        twin_state_status=twin_state.state_status,
        total_count=len(rec_items),
        generated_at=rec_output.get("generated_at", ""),
        recommendations=rec_items,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. POST /buildings/{building_id}/simulate
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/{building_id}/simulate",
    response_model=SimulationResponse,
    summary="Run What-If Energy Demand Simulation",
    description="Runs a what-if scenario simulation (ENERGY_MULTIPLIER, ENERGY_REDUCTION, FIXED_KWH_ADDITION) on the 24-hour forecast baseline without modifying underlying data or models.",
)
def simulate_building_demand(
    building_id: int,
    request: SimulationRequest,
    loader: DataLoader = Depends(get_data_loader),
    forecaster: EnergyForecaster = Depends(get_forecaster),
    sim_engine: EnergySimulationEngine = Depends(get_simulation_engine),
) -> SimulationResponse:
    """Run what-if scenario simulation on 24-hour energy forecast."""
    meter = request.meter
    latest_ts = validate_building_and_meter(loader, building_id=building_id, meter=meter)

    # Resolve cutoff
    if request.cutoff_time is not None:
        try:
            cutoff = pd.to_datetime(request.cutoff_time)
            if pd.isna(cutoff):
                raise ValueError("Invalid timestamp")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid cutoff_time format: '{request.cutoff_time}'.",
            )
    else:
        cutoff = latest_ts - pd.Timedelta(hours=24)

    # 1. Generate baseline forecast
    try:
        baseline_forecast_df = forecaster.forecast_24h(
            building_id=building_id,
            meter=meter,
            cutoff_time=cutoff,
            loader=loader,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Baseline forecast unavailable for simulation: {sanitize_error_message(str(val_err))}",
        )
    except Exception as exc:
        logger.error("Baseline forecast generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during baseline forecast generation.",
        )

    # 2. Run simulation engine
    try:
        sim_res = sim_engine.simulate(
            forecast_df=baseline_forecast_df,
            scenario_type=request.scenario_type,
            value=request.value,
            tariff_per_kwh=request.tariff_per_kwh,
            carbon_intensity_kg_per_kwh=request.carbon_intensity_kg_per_kwh,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(str(val_err)),
        )
    except Exception as exc:
        logger.error("Simulation run error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation failed to execute.",
        )

    # 3. Format hourly items
    hourly_df = sim_res["hourly_df"]
    hourly_items: List[SimulationHourlyItem] = []
    for _, row in hourly_df.iterrows():
        hourly_items.append(
            SimulationHourlyItem(
                timestamp=to_iso_string(row["timestamp"]) or "",
                baseline_kwh=float(row["baseline_kwh"]),
                simulated_kwh=float(row["simulated_kwh"]),
                delta_kwh=float(row["delta_kwh"]),
                delta_percent=float(row["delta_percent"]),
            )
        )

    # 4. Format summary
    summary = SimulationSummary(
        baseline_total_kwh=float(sim_res["baseline_total_kwh"]),
        simulated_total_kwh=float(sim_res["simulated_total_kwh"]),
        delta_total_kwh=float(sim_res["delta_total_kwh"]),
        delta_total_percent=float(sim_res["delta_total_percent"]),
        baseline_peak_kwh=float(sim_res["baseline_peak_kwh"]),
        simulated_peak_kwh=float(sim_res["simulated_peak_kwh"]),
        baseline_peak_hour=to_iso_string(sim_res["baseline_peak_hour"]),
        simulated_peak_hour=to_iso_string(sim_res["simulated_peak_hour"]),
        baseline_total_cost=float(sim_res["baseline_total_cost"]),
        simulated_total_cost=float(sim_res["simulated_total_cost"]),
        delta_total_cost=float(sim_res["delta_total_cost"]),
        baseline_total_carbon_kg=float(sim_res["baseline_total_carbon_kg"]),
        simulated_total_carbon_kg=float(sim_res["simulated_total_carbon_kg"]),
        delta_total_carbon_kg=float(sim_res["delta_total_carbon_kg"]),
        tariff_per_kwh=float(sim_res["tariff_per_kwh"]),
        carbon_intensity_kg_per_kwh=float(sim_res["carbon_intensity_kg_per_kwh"]),
    )

    return SimulationResponse(
        building_id=building_id,
        meter=meter,
        scenario_type=request.scenario_type,
        scenario_value=request.value,
        summary=summary,
        hourly_simulation=hourly_items,
    )
