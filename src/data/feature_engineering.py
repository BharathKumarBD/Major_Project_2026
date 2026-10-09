"""Feature engineering pipeline for Twinergy 2.0.

Provides modular, reproducible functions to compute calendar features, autoregressive lags,
rolling statistics, weather interactions, and the heuristic occupancy proxy without target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import (
    ENGINEERED_INTERACTION_FEATURES,
    ENGINEERED_LAG_FEATURES,
    ENGINEERED_ROLLING_FEATURES,
    ENGINEERED_TIME_FEATURES,
    LIGHTGBM_FEATURES,
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate calendar and cyclical time features from timestamp.
    
    Features created:
    - hour (0-23)
    - dayofweek (0-6)
    - month (1-12)
    - dayofyear (1-366)
    - is_weekend (0 or 1)
    - is_business_hours (1 if weekday 08:00-18:00, else 0)
    - hour_sin, hour_cos (24-hour cycle)
    - month_sin, month_cos (12-month cycle)
    """
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        
    ts = out["timestamp"]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    
    # Weekday 08:00 to 18:00 (hour 8 through 17)
    out["is_business_hours"] = (
        (out["is_weekend"] == 0) & (out["hour"] >= 8) & (out["hour"] < 18)
    ).astype(int)
    
    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    
    return out


def add_autoregressive_lags(
    df: pd.DataFrame,
    target_col: str = "meter_reading_log",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate autoregressive lag features: lag_1hr, lag_24hr, and lag_168hr.
    
    Lags are computed grouped by building_id and meter, sorted chronologically.
    Initial missing lag_168hr values are imputed using rolling_mean_24hr.
    """
    if group_cols is None:
        group_cols = ["building_id", "meter"] if "meter" in df.columns else ["building_id"]
        
    out = df.copy().sort_values(group_cols + ["timestamp"]).reset_index(drop=True)
    
    grouped = out.groupby(group_cols)[target_col]
    out["lag_1hr"] = grouped.shift(1)
    out["lag_24hr"] = grouped.shift(24)
    out["lag_168hr"] = grouped.shift(168)
    
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "meter_reading_log",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate rolling mean and standard deviation features strictly without target leakage.
    
    A 1-step shift is applied prior to computing rolling windows, ensuring the rolling
    window strictly covers observations up to t-1.
    
    Features created:
    - rolling_mean_6hr
    - rolling_mean_24hr
    - rolling_std_24hr
    - rolling_mean_168hr
    """
    if group_cols is None:
        group_cols = ["building_id", "meter"] if "meter" in df.columns else ["building_id"]
        
    out = df.copy().sort_values(group_cols + ["timestamp"]).reset_index(drop=True)
    
    # 1-step shifted series per series group
    shifted = out.groupby(group_cols)[target_col].shift(1)
    
    # Compute rolling windows on shifted series grouped by building/meter
    grp_shifted = shifted.groupby([out[col] for col in group_cols])
    out["rolling_mean_6hr"] = grp_shifted.transform(lambda x: x.rolling(6, min_periods=1).mean())
    out["rolling_mean_24hr"] = grp_shifted.transform(lambda x: x.rolling(24, min_periods=1).mean())
    out["rolling_std_24hr"] = grp_shifted.transform(lambda x: x.rolling(24, min_periods=1).std()).fillna(0.0)
    out["rolling_mean_168hr"] = grp_shifted.transform(lambda x: x.rolling(168, min_periods=1).mean())
    
    # Backfill lag_168hr with rolling_mean_24hr if lag_168hr exists
    if "lag_168hr" in out.columns:
        out["lag_168hr"] = out["lag_168hr"].fillna(out["rolling_mean_24hr"])
        
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate interaction features and heuristic occupancy proxy.
    
    Features created:
    - temp_x_hour = air_temperature * hour
    - temp_x_weekend = air_temperature * is_weekend
    - occupancy_proxy = is_business_hours * rolling_mean_6hr (DISCLOSED HEURISTIC PROXY)
    """
    out = df.copy()
    
    temp = out["air_temperature"].fillna(out["air_temperature"].median() if "air_temperature" in out.columns else 20.0)
    out["temp_x_hour"] = temp * out["hour"]
    out["temp_x_weekend"] = temp * out["is_weekend"]
    
    r6 = out["rolling_mean_6hr"] if "rolling_mean_6hr" in out.columns else 0.0
    out["occupancy_proxy"] = out["is_business_hours"] * r6
    
    return out


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete end-to-end feature engineering pipeline."""
    df_feat = add_calendar_features(df)
    df_feat = add_autoregressive_lags(df_feat)
    df_feat = add_rolling_features(df_feat)
    df_feat = add_interaction_features(df_feat)
    return df_feat
