"""Data preprocessing pipeline for Twinergy 2.0.

Provides modular, reproducible functions to clean raw ASHRAE energy and weather data,
handle missing values, filter physical outliers, and align data contracts.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.schema import (
    BUILDING_METADATA_COLUMNS,
    IDENTIFIER_COLUMNS,
    WEATHER_COLUMNS,
)


def clean_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing weather observations by site via forward-fill and backward-fill.
    
    Args:
        weather_df: Raw weather DataFrame containing 'site_id', 'timestamp', and weather metrics.
        
    Returns:
        Cleaned weather DataFrame with core continuous features imputed.
    """
    df = weather_df.copy()
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    # Sort chronologically by site
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    
    core_weather = ["air_temperature", "dew_temperature", "wind_speed", "sea_level_pressure"]
    for col in core_weather:
        if col in df.columns:
            df[col] = df.groupby("site_id")[col].transform(lambda x: x.ffill().bfill())
            # If still null for an entire site, fill with global median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
    return df


def filter_energy_outliers(
    df: pd.DataFrame,
    reading_col: str = "meter_reading",
    quantile_cutoff: float = 0.999,
) -> pd.DataFrame:
    """Remove non-positive readings and extreme upper-percentile anomalies per meter type.
    
    Args:
        df: Input DataFrame containing meter readings.
        reading_col: Column name for raw meter reading.
        quantile_cutoff: Upper quantile threshold (default 99.9th percentile).
        
    Returns:
        Filtered DataFrame.
    """
    out = df[df[reading_col] > 0].copy()
    
    # Filter top quantile per meter type if meter column exists
    if "meter" in out.columns:
        valid_indices = []
        for meter_id, group in out.groupby("meter"):
            upper_limit = group[reading_col].quantile(quantile_cutoff)
            valid_indices.extend(group[group[reading_col] <= upper_limit].index)
        out = out.loc[valid_indices].sort_index()
    else:
        upper_limit = out[reading_col].quantile(quantile_cutoff)
        out = out[out[reading_col] <= upper_limit]
        
    return out


def merge_ashrae_sources(
    train_df: pd.DataFrame,
    building_meta_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge train energy readings, building metadata, and cleaned weather observations.
    
    Args:
        train_df: Raw or filtered meter readings DataFrame.
        building_meta_df: Building metadata DataFrame.
        weather_df: Weather observations DataFrame.
        
    Returns:
        Merged unified DataFrame.
    """
    # Merge building metadata
    df = train_df.merge(building_meta_df, on="building_id", how="left")
    
    # Ensure timestamps are parsed
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "timestamp" in weather_df.columns and not pd.api.types.is_datetime64_any_dtype(weather_df["timestamp"]):
        weather_df = weather_df.copy()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        
    # Merge weather observations
    df = df.merge(weather_df, on=["site_id", "timestamp"], how="left")
    
    # Add target log scale: ln(1 + meter_reading)
    if "meter_reading" in df.columns:
        df["meter_reading_log"] = np.log1p(df["meter_reading"])
        
    return df
