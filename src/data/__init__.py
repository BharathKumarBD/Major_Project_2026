"""Twinergy 2.0 Data Foundation Package.

Provides schema definitions, high-performance data loading (Parquet + DuckDB),
and automated data contract validation.
"""

from src.data.loader import DataLoader, load_data, resolve_data_path
from src.data.schema import (
    ALL_COLUMNS,
    ANOMALY_FEATURES,
    BUILDING_METADATA_COLUMNS,
    COLUMN_DTYPES,
    ENGINEERED_INTERACTION_FEATURES,
    ENGINEERED_LAG_FEATURES,
    ENGINEERED_ROLLING_FEATURES,
    ENGINEERED_TIME_FEATURES,
    IDENTIFIER_COLUMNS,
    LIGHTGBM_FEATURES,
    METER_TYPE_NAMES,
    OBSERVED_ENERGY_FIELD,
    PRIMARY_FORECAST_TARGET,
    TARGET_COLUMNS,
    WEATHER_COLUMNS,
)
from src.data.validation import (
    ValidationResult,
    ValidationSuiteResult,
    run_all_validations,
    validate_anomaly_features,
    validate_building_ids,
    validate_duplicates,
    validate_lightgbm_features,
    validate_meters,
    validate_missing_values,
    validate_required_columns,
    validate_target,
    validate_timestamps,
)

__all__ = [
    # Loader
    "DataLoader",
    "load_data",
    "resolve_data_path",
    # Schema
    "ALL_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "BUILDING_METADATA_COLUMNS",
    "WEATHER_COLUMNS",
    "ENGINEERED_TIME_FEATURES",
    "ENGINEERED_LAG_FEATURES",
    "ENGINEERED_ROLLING_FEATURES",
    "ENGINEERED_INTERACTION_FEATURES",
    "TARGET_COLUMNS",
    "LIGHTGBM_FEATURES",
    "ANOMALY_FEATURES",
    "PRIMARY_FORECAST_TARGET",
    "OBSERVED_ENERGY_FIELD",
    "METER_TYPE_NAMES",
    "COLUMN_DTYPES",
    # Validation
    "ValidationResult",
    "ValidationSuiteResult",
    "run_all_validations",
    "validate_required_columns",
    "validate_timestamps",
    "validate_building_ids",
    "validate_meters",
    "validate_target",
    "validate_lightgbm_features",
    "validate_anomaly_features",
    "validate_missing_values",
    "validate_duplicates",
]
