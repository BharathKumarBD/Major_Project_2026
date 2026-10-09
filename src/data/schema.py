"""Data schema and contract definitions for Twinergy 2.0.

Defines column names, data types, model feature contracts, and metadata
for the ASHRAE Great Energy Predictor III cleaned dataset.
"""

from typing import Dict, List, Mapping

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN GROUPINGS
# ─────────────────────────────────────────────────────────────────────────────

IDENTIFIER_COLUMNS: List[str] = [
    "building_id",
    "meter",
    "timestamp",
]

BUILDING_METADATA_COLUMNS: List[str] = [
    "site_id",
    "primary_use",
    "square_feet",
    "year_built",
    "floor_count",
]

WEATHER_COLUMNS: List[str] = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

ENGINEERED_TIME_FEATURES: List[str] = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "is_business_hours",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

ENGINEERED_LAG_FEATURES: List[str] = [
    "lag_1hr",
    "lag_24hr",
    "lag_168hr",
]

ENGINEERED_ROLLING_FEATURES: List[str] = [
    "rolling_mean_6hr",
    "rolling_mean_24hr",
    "rolling_std_24hr",
    "rolling_mean_168hr",
]

ENGINEERED_INTERACTION_FEATURES: List[str] = [
    "temp_x_hour",
    "temp_x_weekend",
    "occupancy_proxy",
]

TARGET_COLUMNS: List[str] = [
    "meter_reading",      # Raw observed consumption (kWh / native units)
    "meter_reading_log",  # Target used for model training: log1p(meter_reading)
]

ALL_COLUMNS: List[str] = [
    "building_id",
    "meter",
    "timestamp",
    "meter_reading",
    "site_id",
    "primary_use",
    "square_feet",
    "year_built",
    "floor_count",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "meter_reading_log",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "is_business_hours",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1hr",
    "lag_24hr",
    "lag_168hr",
    "rolling_mean_6hr",
    "rolling_mean_24hr",
    "rolling_std_24hr",
    "rolling_mean_168hr",
    "temp_x_hour",
    "temp_x_weekend",
    "occupancy_proxy",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURE CONTRACTS (Must strictly preserve compatibility)
# ─────────────────────────────────────────────────────────────────────────────

# Exact 22 features required by twinergy_lgbm.pkl
LIGHTGBM_FEATURES: List[str] = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "is_business_hours",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1hr",
    "lag_24hr",
    "lag_168hr",
    "rolling_mean_6hr",
    "rolling_mean_24hr",
    "rolling_std_24hr",
    "rolling_mean_168hr",
    "temp_x_hour",
    "temp_x_weekend",
    "occupancy_proxy",
    "air_temperature",
    "square_feet",
]

# Exact 8 features required by twinergy_isoforest.pkl
ANOMALY_FEATURES: List[str] = [
    "meter_reading_log",
    "rolling_mean_6hr",
    "rolling_std_24hr",
    "lag_1hr",
    "lag_24hr",
    "hour",
    "is_weekend",
    "air_temperature",
]

# Primary model targets
PRIMARY_FORECAST_TARGET: str = "meter_reading_log"
OBSERVED_ENERGY_FIELD: str = "meter_reading"

# ─────────────────────────────────────────────────────────────────────────────
# METER TYPE MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

METER_TYPE_NAMES: Mapping[int, str] = {
    0: "Electricity",
    1: "Chilled Water",
    2: "Steam",
    3: "Hot Water",
}

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_DTYPES: Dict[str, str] = {
    "building_id": "int64",
    "meter": "int64",
    "timestamp": "datetime64[ns]",
    "meter_reading": "float64",
    "site_id": "int64",
    "primary_use": "string",
    "square_feet": "int64",
    "year_built": "float64",
    "floor_count": "string",
    "air_temperature": "float64",
    "cloud_coverage": "float64",
    "dew_temperature": "float64",
    "precip_depth_1_hr": "float64",
    "sea_level_pressure": "float64",
    "wind_direction": "float64",
    "wind_speed": "float64",
    "meter_reading_log": "float64",
    "hour": "int64",
    "dayofweek": "int64",
    "month": "int64",
    "dayofyear": "int64",
    "is_weekend": "int64",
    "is_business_hours": "int64",
    "hour_sin": "float64",
    "hour_cos": "float64",
    "month_sin": "float64",
    "month_cos": "float64",
    "lag_1hr": "float64",
    "lag_24hr": "float64",
    "lag_168hr": "float64",
    "rolling_mean_6hr": "float64",
    "rolling_mean_24hr": "float64",
    "rolling_std_24hr": "float64",
    "rolling_mean_168hr": "float64",
    "temp_x_hour": "float64",
    "temp_x_weekend": "float64",
    "occupancy_proxy": "float64",
}
