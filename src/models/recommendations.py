"""Deterministic Recommendation Engine for Twinergy 2.0.

Provides explainable, non-causal energy management recommendations derived
from TwinState software representations, anomaly classifications, forecasts,
and what-if simulation outputs.

NOTE: Recommendations are rule-based advisory guidelines using language like
'investigate', 'check', and 'consider'. They do NOT establish physical causality
or claim automated control capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.models.twin_state import TwinState

PRIORITY_ORDER = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
    "INFO": 5,
}

VALID_PRIORITIES = set(PRIORITY_ORDER.keys())
VALID_CATEGORIES = {"ENERGY", "ANOMALY", "COST", "CARBON", "OPERATIONS"}


@dataclass
class Recommendation:
    """Structured representation of a single energy management recommendation.

    Attributes:
        recommendation_id: Unique identifier string for recommendation.
        priority: Recommendation priority ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO').
        category: Functional category ('ENERGY', 'ANOMALY', 'COST', 'CARBON', 'OPERATIONS').
        title: Short title summarizing the recommendation.
        action: Clear advisory action item (using non-causal verbs like 'review', 'investigate').
        reason: Explanation of why this recommendation was triggered.
        evidence: Data-backed evidence string referencing actual TwinState / forecast values.
    """

    recommendation_id: str
    priority: str
    category: str
    title: str
    action: str
    reason: str
    evidence: str

    def __post_init__(self):
        """Validate recommendation attributes."""
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{self.priority}'. Must be one of {sorted(list(VALID_PRIORITIES))}"
            )
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. Must be one of {sorted(list(VALID_CATEGORIES))}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Recommendation object to a clean, JSON-serializable dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "priority": self.priority,
            "category": self.category,
            "title": self.title,
            "action": self.action,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class RecommendationEngine:
    """Service layer for constructing rule-based, deterministic recommendations."""

    DEFAULT_COST_THRESHOLD: float = 10000.0  # ₹ 10,000 engineering default threshold
    DEFAULT_CARBON_THRESHOLD: float = 1000.0  # 1,000 kg CO2e engineering default threshold

    def __init__(
        self,
        cost_threshold: float = DEFAULT_COST_THRESHOLD,
        carbon_threshold: float = DEFAULT_CARBON_THRESHOLD,
    ):
        """Initialize RecommendationEngine with configurable threshold parameters.

        Args:
            cost_threshold: Configurable forecast 24h cost threshold in ₹ (default 10000.0).
            carbon_threshold: Configurable forecast 24h carbon threshold in kg CO2e (default 1000.0).
        """
        self.cost_threshold = self._validate_threshold(cost_threshold, "cost_threshold")
        self.carbon_threshold = self._validate_threshold(carbon_threshold, "carbon_threshold")

    def _validate_threshold(self, val: Any, name: str) -> float:
        """Validate numeric threshold parameters."""
        if isinstance(val, (bool, str)) and not isinstance(val, (int, float)):
            try:
                val = float(val)
            except (ValueError, TypeError):
                raise TypeError(f"{name} must be numeric, got {type(val)}")

        if not isinstance(val, (int, float, np.number)):
            raise TypeError(f"{name} must be numeric, got {type(val)}")

        f_val = float(val)
        if math.isnan(f_val) or np.isnan(f_val):
            raise ValueError(f"{name} cannot be NaN.")
        if math.isinf(f_val) or np.isinf(f_val):
            raise ValueError(f"{name} cannot be infinite.")
        if f_val < 0.0:
            raise ValueError(f"{name} cannot be negative, got {f_val}")

        return f_val

    def generate_recommendations(
        self,
        twin_state: TwinState,
        simulation_result: Optional[Dict[str, Any]] = None,
        cost_threshold: Optional[float] = None,
        carbon_threshold: Optional[float] = None,
    ) -> List[Recommendation]:
        """Generate deterministic recommendations from TwinState and optional simulation results.

        Args:
            twin_state: Input TwinState software representation object.
            simulation_result: Optional dictionary output from EnergySimulationEngine.simulate.
            cost_threshold: Optional runtime override for cost threshold.
            carbon_threshold: Optional runtime override for carbon threshold.

        Returns:
            List of Recommendation objects ordered by priority (CRITICAL -> HIGH -> MEDIUM -> LOW -> INFO).
        """
        if not isinstance(twin_state, TwinState):
            raise TypeError(f"twin_state must be a TwinState object, got {type(twin_state)}")

        c_thresh = (
            self.cost_threshold
            if cost_threshold is None
            else self._validate_threshold(cost_threshold, "cost_threshold")
        )
        cb_thresh = (
            self.carbon_threshold
            if carbon_threshold is None
            else self._validate_threshold(carbon_threshold, "carbon_threshold")
        )

        recs: List[Recommendation] = []
        rec_id_counter = 1

        def _make_id(prefix: str) -> str:
            nonlocal rec_id_counter
            rid = f"REC-{prefix}-{rec_id_counter:03d}"
            rec_id_counter += 1
            return rid

        # Rule 1 — Critical Anomaly
        if twin_state.severity == "CRITICAL":
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            dev = (
                twin_state.deviation_ratio * 100.0
                if twin_state.deviation_ratio is not None
                else 0.0
            )
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("CRIT"),
                    priority="CRITICAL",
                    category="ANOMALY",
                    title="Investigate Critical Energy Deviation",
                    action="Review current building loads and operating conditions immediately.",
                    reason="Energy consumption is substantially above or below the expected level.",
                    evidence=(
                        f"Building {twin_state.building_id} severity is CRITICAL at {twin_state.timestamp} "
                        f"(Actual: {act:.1f} kWh, Expected: {exp:.1f} kWh, Deviation: {dev:+.1f}%)."
                    ),
                )
            )

        # Rule 2 — High Demand Anomaly
        if twin_state.anomaly_type == "HIGH_DEMAND":
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            dev = (
                twin_state.deviation_ratio * 100.0
                if twin_state.deviation_ratio is not None
                else 0.0
            )
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("ANOM"),
                    priority="HIGH",
                    category="ANOMALY",
                    title="Investigate Elevated Energy Demand",
                    action="Inspect building HVAC and major electrical equipment schedules.",
                    reason="Actual consumption significantly exceeded expected baseline.",
                    evidence=(
                        f"HIGH_DEMAND anomaly detected at {twin_state.timestamp}: actual consumption "
                        f"({act:.1f} kWh) is {dev:+.1f}% above expected ({exp:.1f} kWh)."
                    ),
                )
            )

        # Rule 3 — Rapid Spike Anomaly
        if twin_state.anomaly_type == "RAPID_SPIKE":
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("SPIKE"),
                    priority="HIGH",
                    category="ANOMALY",
                    title="Investigate Rapid Consumption Spike",
                    action="Check for sudden load changes, process shifts, or unexpected equipment operations.",
                    reason="Sharp consumption surge detected relative to recent rolling baseline.",
                    evidence=(
                        f"RAPID_SPIKE anomaly detected at {twin_state.timestamp} (Actual: {act:.1f} kWh, "
                        f"Expected baseline: {exp:.1f} kWh)."
                    ),
                )
            )

        # Rule 4 — Persistent High Demand Anomaly
        if twin_state.anomaly_type == "PERSISTENT_HIGH":
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("PERS"),
                    priority="HIGH",
                    category="OPERATIONS",
                    title="Review Sustained High Demand",
                    action="Review continuous baseline loads and operating schedules across consecutive hours.",
                    reason="Elevated energy demand sustained over multiple consecutive hours.",
                    evidence=(
                        f"PERSISTENT_HIGH demand pattern active at {twin_state.timestamp} "
                        f"(Actual: {act:.1f} kWh, Expected: {exp:.1f} kWh)."
                    ),
                )
            )

        # Rule 5 — Low Demand Anomaly
        if twin_state.anomaly_type == "LOW_DEMAND":
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            dev = (
                twin_state.deviation_ratio * 100.0
                if twin_state.deviation_ratio is not None
                else 0.0
            )
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("LOW"),
                    priority="MEDIUM",
                    category="OPERATIONS",
                    title="Check Reduced Demand Activity",
                    action="Verify if low consumption is planned or indicates inactive systems or sensor issue.",
                    reason="Energy consumption is significantly below expected operating level.",
                    evidence=(
                        f"LOW_DEMAND anomaly detected at {twin_state.timestamp}: actual consumption "
                        f"({act:.1f} kWh) is {abs(dev):.1f}% below expected ({exp:.1f} kWh)."
                    ),
                )
            )

        # Rule 6 — High Forecast Cost
        if twin_state.forecast_total_cost is not None and twin_state.forecast_total_cost > c_thresh:
            fc_cost = twin_state.forecast_total_cost
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("COST"),
                    priority="MEDIUM",
                    category="COST",
                    title="Consider Demand Management for High Cost Forecast",
                    action="Evaluate load-shifting options to reduce peak-period energy charges.",
                    reason="Projected 24-hour cost exceeds the configured cost optimization threshold.",
                    evidence=(
                        f"Projected 24h cost is ₹{fc_cost:,.2f}, exceeding the engineering assumption threshold "
                        f"of ₹{c_thresh:,.2f}."
                    ),
                )
            )

        # Rule 7 — High Forecast Carbon
        if (
            twin_state.forecast_total_carbon_kg is not None
            and twin_state.forecast_total_carbon_kg > cb_thresh
        ):
            fc_carbon = twin_state.forecast_total_carbon_kg
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("CARB"),
                    priority="MEDIUM",
                    category="CARBON",
                    title="Review Carbon Emission Impact",
                    action="Consider energy conservation measures during peak carbon intensity periods.",
                    reason="Projected 24-hour carbon emissions exceed configured carbon target threshold.",
                    evidence=(
                        f"Projected 24h carbon emissions are {fc_carbon:,.2f} kg CO2e, exceeding configured "
                        f"target threshold of {cb_thresh:,.2f} kg CO2e."
                    ),
                )
            )

        # Rule 8 — Optional Simulation Context Opportunity
        if simulation_result is not None:
            delta_cost = simulation_result.get("delta_total_cost", 0.0)
            delta_kwh = simulation_result.get("delta_total_kwh", 0.0)
            scen_type = simulation_result.get("scenario_type", "SIMULATION")
            scen_val = simulation_result.get("scenario_value", 0.0)

            if delta_kwh < 0.0 or delta_cost < 0.0:
                recs.append(
                    Recommendation(
                        recommendation_id=_make_id("SIM"),
                        priority="LOW",
                        category="ENERGY",
                        title="Simulated Scenario Savings Opportunity",
                        action="Consider evaluating the active what-if simulation scenario for potential implementation.",
                        reason="Scenario estimate indicates potential energy and cost savings relative to baseline.",
                        evidence=(
                            f"Scenario '{scen_type}' (value: {scen_val}) shows an estimated reduction of "
                            f"{abs(delta_kwh):.1f} kWh and ₹{abs(delta_cost):,.2f} cost savings (scenario estimate)."
                        ),
                    )
                )

        # Rule 9 — Normal State Fallback
        if twin_state.state_status == "NORMAL" and len(recs) == 0:
            act = twin_state.actual_kwh if twin_state.actual_kwh is not None else 0.0
            exp = twin_state.expected_kwh if twin_state.expected_kwh is not None else 0.0
            recs.append(
                Recommendation(
                    recommendation_id=_make_id("NORM"),
                    priority="INFO",
                    category="ENERGY",
                    title="Energy System Nominal",
                    action="Continue standard building energy monitoring.",
                    reason="No significant energy consumption anomaly detected.",
                    evidence=(
                        f"Building {twin_state.building_id} state status is NORMAL at {twin_state.timestamp} "
                        f"(Actual: {act:.1f} kWh, Expected: {exp:.1f} kWh)."
                    ),
                )
            )

        # Deduplicate recommendations by (category, title, action) preserving highest priority
        deduped: List[Recommendation] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        recs_sorted = sorted(recs, key=lambda r: PRIORITY_ORDER.get(r.priority, 99))

        for r in recs_sorted:
            key = (r.category, r.title, r.action)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(r)

        return deduped

    def recommend(
        self,
        twin_state: TwinState,
        simulation_result: Optional[Dict[str, Any]] = None,
        cost_threshold: Optional[float] = None,
        carbon_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience method returning structured dictionary response with recommendations list."""
        recs = self.generate_recommendations(
            twin_state=twin_state,
            simulation_result=simulation_result,
            cost_threshold=cost_threshold,
            carbon_threshold=carbon_threshold,
        )
        return {
            "recommendations": [r.to_dict() for r in recs],
            "total_count": len(recs),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


def load_recommendation_engine(
    cost_threshold: float = RecommendationEngine.DEFAULT_COST_THRESHOLD,
    carbon_threshold: float = RecommendationEngine.DEFAULT_CARBON_THRESHOLD,
) -> RecommendationEngine:
    """Convenience factory function to instantiate RecommendationEngine."""
    return RecommendationEngine(
        cost_threshold=cost_threshold,
        carbon_threshold=carbon_threshold,
    )
