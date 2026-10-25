"""Regression tests for building month data availability, KPI calculations, and display handling."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.schema import LIGHTGBM_FEATURES
from src.models.forecasting import load_forecaster
from src.models.explainability import load_explainer
from src.models.impact import EnergyImpactCalculator


def test_building_0_month_availability():
    """Building 0 (Site 0) has clean observations starting in May (month 5) through Dec."""
    loader = DataLoader()
    b0_df = loader.load(building_id=0, meter=0)
    assert len(b0_df) > 0, "Building 0 data must be loadable"

    available_months = sorted(b0_df["timestamp"].dt.month.unique().tolist())
    # Verify January-April are empty for Building 0 in ASHRAE Site 0
    for m in [1, 2, 3, 4]:
        m_slice = b0_df[b0_df["timestamp"].dt.month == m]
        assert len(m_slice) == 0, f"Building 0 should have no records for month {m}"

    # Verify May-December have records
    for m in [5, 6, 7, 8, 9, 10, 11, 12]:
        assert m in available_months, f"Building 0 must have records for month {m}"


def test_building_kpi_and_forecast_on_populated_month():
    """Populated month (Building 0, May) computes all 5 KPI metrics and forecast curves."""
    loader = DataLoader()
    forecaster = load_forecaster()
    impact_calc = EnergyImpactCalculator()

    b0_df = loader.load(building_id=0, meter=0)
    may_df = b0_df[b0_df["timestamp"].dt.month == 5].copy()
    assert len(may_df) > 0

    clean_may = may_df.dropna(subset=LIGHTGBM_FEATURES).copy()
    assert len(clean_may) > 0

    clean_may["predicted"] = forecaster.predict(clean_may, return_kwh=True)
    clean_may["actual"] = np.expm1(clean_may["meter_reading_log"])

    # 1. Current kWh
    current_kwh = clean_may["actual"].iloc[-1]
    assert np.isfinite(current_kwh) and current_kwh >= 0

    # 2. Predicted kWh
    predicted_kwh = clean_may["predicted"].iloc[-1]
    assert np.isfinite(predicted_kwh) and predicted_kwh >= 0

    # 3. Monthly avg kWh
    avg_kwh = clean_may["actual"].mean()
    assert np.isfinite(avg_kwh) and avg_kwh > 0

    # 4. Total kWh and CO2 estimate
    total_kwh = clean_may["actual"].sum()
    co2_kg = total_kwh * impact_calc.carbon_intensity_kg_per_kwh
    assert np.isfinite(co2_kg) and co2_kg > 0


def test_building_20_january_availability():
    """Building 20 has observations in January (month 1)."""
    loader = DataLoader()
    forecaster = load_forecaster()

    b20_df = loader.load(building_id=20, meter=0)
    jan_df = b20_df[b20_df["timestamp"].dt.month == 1].copy()
    assert len(jan_df) > 0, "Building 20 must have January data"

    clean_jan = jan_df.dropna(subset=LIGHTGBM_FEATURES).copy()
    assert len(clean_jan) > 0
    clean_jan["predicted"] = forecaster.predict(clean_jan, return_kwh=True)
    clean_jan["actual"] = np.expm1(clean_jan["meter_reading_log"])

    assert clean_jan["actual"].mean() > 0
    assert clean_jan["predicted"].mean() > 0


def test_global_feature_importance_independent_of_month_data():
    """Global LightGBM tree importance is accessible without requiring non-empty month observations."""
    explainer = load_explainer()
    feat_imp = explainer.get_tree_feature_importance(importance_type="gain")

    assert isinstance(feat_imp, pd.DataFrame)
    assert len(feat_imp) == len(LIGHTGBM_FEATURES)
    assert set(feat_imp.columns) == {"feature", "importance", "importance_type"}
    assert (feat_imp["importance"] >= 0).all()
