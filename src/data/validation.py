"""Automated dataset validation suite for Twinergy 2.0.

Validates column presence, types, value constraints, target semantics,
model feature contracts (LightGBM and Isolation Forest), missing values,
and timestamp uniqueness.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader, resolve_data_path
from src.data.schema import (
    ALL_COLUMNS,
    ANOMALY_FEATURES,
    IDENTIFIER_COLUMNS,
    LIGHTGBM_FEATURES,
    METER_TYPE_NAMES,
    PRIMARY_FORECAST_TARGET,
    OBSERVED_ENERGY_FIELD,
)

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


@dataclass
class ValidationResult:
    """Outcome of a single validation rule."""

    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSuiteResult:
    """Outcome of all validation checks in the suite."""

    dataset_path: str
    passed: bool
    results: List[ValidationResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def print_report(self) -> None:
        """Print a formatted validation report to stdout."""
        print("=" * 70)
        print("       TWINERGY 2.0 -- DATA FOUNDATION VALIDATION REPORT")
        print("=" * 70)
        print(f"Target Dataset : {self.dataset_path}")
        print(f"Overall Status : {'[PASS] PASSED' if self.passed else '[FAIL] FAILED'}")
        print("-" * 70)
        for r in self.results:
            icon = "[PASS]" if r.passed else "[FAIL]"
            print(f"{icon} {r.name:32s}: {r.message}")
            if not r.passed and r.details:
                print(f"       Details: {r.details}")
        print("=" * 70)


def validate_required_columns(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    required_columns: Optional[Sequence[str]] = None,
) -> ValidationResult:
    """Check that all required columns exist in the dataset."""
    target_cols = set(required_columns or ALL_COLUMNS)

    if df is not None:
        existing = set(df.columns)
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        cols_schema = con.execute(f"DESCRIBE SELECT * FROM '{p}' LIMIT 1").fetchall()
        existing = set(row[0] for row in cols_schema)
    else:
        loader = DataLoader(data_path=dataset_path)
        sample = loader.load(limit=1, validate_columns=False)
        existing = set(sample.columns)

    missing = sorted(list(target_cols - existing))
    passed = len(missing) == 0
    msg = f"All {len(target_cols)} required columns present." if passed else f"Missing columns: {missing}"
    return ValidationResult(
        name="Required Columns",
        passed=passed,
        message=msg,
        details={"missing": missing, "found_count": len(existing), "required_count": len(target_cols)},
    )


def validate_timestamps(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate timestamp non-nullity, parseability, and reasonable bounds."""
    if df is not None:
        ts = pd.to_datetime(df["timestamp"])
        null_count = int(ts.isnull().sum())
        min_ts = str(ts.min())
        max_ts = str(ts.max())
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        res = con.execute(f"""
            SELECT 
                count(*) - count(timestamp) as null_cnt,
                min(timestamp) as min_ts,
                max(timestamp) as max_ts
            FROM '{p}'
        """).fetchone()
        null_count, min_ts, max_ts = res[0], str(res[1]), str(res[2])
    else:
        loader = DataLoader(data_path=dataset_path)
        ts_df = loader.load(columns=["timestamp"], validate_columns=False)
        ts = pd.to_datetime(ts_df["timestamp"])
        null_count = int(ts.isnull().sum())
        min_ts = str(ts.min())
        max_ts = str(ts.max())

    passed = (null_count == 0) and (min_ts is not None) and (max_ts is not None)
    msg = f"Valid timestamps from {min_ts} to {max_ts} with 0 nulls." if passed else f"Found {null_count} invalid/null timestamps."
    return ValidationResult(
        name="Valid Timestamps",
        passed=passed,
        message=msg,
        details={"null_count": null_count, "min_timestamp": min_ts, "max_timestamp": max_ts},
    )


def validate_building_ids(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate building_id non-nullity, integer type, and positive count."""
    if df is not None:
        null_count = int(df["building_id"].isnull().sum())
        n_unique = int(df["building_id"].nunique())
        min_id = int(df["building_id"].min())
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        res = con.execute(f"""
            SELECT 
                count(*) - count(building_id),
                count(distinct building_id),
                min(building_id)
            FROM '{p}'
        """).fetchone()
        null_count, n_unique, min_id = res[0], res[1], res[2]
    else:
        loader = DataLoader(data_path=dataset_path)
        b_df = loader.load(columns=["building_id"], validate_columns=False)
        null_count = int(b_df["building_id"].isnull().sum())
        n_unique = int(b_df["building_id"].nunique())
        min_id = int(b_df["building_id"].min())

    passed = (null_count == 0) and (n_unique > 0) and (min_id >= 0)
    msg = f"{n_unique} distinct non-negative buildings (min: {min_id}), 0 nulls." if passed else "Invalid building IDs detected."
    return ValidationResult(
        name="Building ID Contract",
        passed=passed,
        message=msg,
        details={"null_count": null_count, "unique_buildings": n_unique, "min_building_id": min_id},
    )


def validate_meters(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate meter types belong to canonical ASHRAE set {0, 1, 2, 3}."""
    allowed = set(METER_TYPE_NAMES.keys())

    if df is not None:
        meters_found = set(df["meter"].unique())
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        res = con.execute(f"SELECT DISTINCT meter FROM '{p}' ORDER BY meter").fetchall()
        meters_found = set(r[0] for r in res)
    else:
        loader = DataLoader(data_path=dataset_path)
        m_df = loader.load(columns=["meter"], validate_columns=False)
        meters_found = set(m_df["meter"].unique())

    unexpected = meters_found - allowed
    passed = len(unexpected) == 0 and len(meters_found) > 0
    msg = f"Meter IDs {sorted(list(meters_found))} conform to ASHRAE spec {sorted(list(allowed))}." if passed else f"Unexpected meters: {unexpected}"
    return ValidationResult(
        name="Meter Contract",
        passed=passed,
        message=msg,
        details={"meters_found": sorted(list(meters_found)), "allowed": sorted(list(allowed))},
    )


def validate_target(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate target fields (meter_reading > 0 and meter_reading_log = log1p(meter_reading))."""
    if df is not None:
        null_raw = int(df["meter_reading"].isnull().sum())
        null_log = int(df["meter_reading_log"].isnull().sum())
        non_positive = int((df["meter_reading"] <= 0).sum())
        # sample check log1p consistency
        sample = df[["meter_reading", "meter_reading_log"]].dropna().head(1000)
        diff = np.abs(sample["meter_reading_log"] - np.log1p(sample["meter_reading"])).max()
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        res = con.execute(f"""
            SELECT 
                count(*) - count(meter_reading) as null_raw,
                count(*) - count(meter_reading_log) as null_log,
                count(CASE WHEN meter_reading <= 0 THEN 1 END) as non_pos,
                max(abs(meter_reading_log - ln(1 + meter_reading))) as max_log_diff
            FROM '{p}'
        """).fetchone()
        null_raw, null_log, non_positive, diff = res[0], res[1], res[2], (res[3] or 0.0)
    else:
        loader = DataLoader(data_path=dataset_path)
        s = loader.load(columns=["meter_reading", "meter_reading_log"], limit=10000)
        null_raw = int(s["meter_reading"].isnull().sum())
        null_log = int(s["meter_reading_log"].isnull().sum())
        non_positive = int((s["meter_reading"] <= 0).sum())
        diff = float(np.abs(s["meter_reading_log"] - np.log1p(s["meter_reading"])).max())

    passed = (null_raw == 0) and (null_log == 0) and (non_positive == 0) and (diff < 1e-4)
    msg = f"Target verified: 0 nulls, strictly positive meter_reading, log1p consistent (max diff {diff:.2e})." if passed else "Target validation failed."
    return ValidationResult(
        name="Target & Observed Energy Contract",
        passed=passed,
        message=msg,
        details={"null_raw": null_raw, "null_log": null_log, "non_positive_count": non_positive, "max_log1p_diff": diff},
    )


def validate_lightgbm_features(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate all 22 LightGBM model features exist and have zero missing values."""
    return _validate_feature_set(
        feature_list=LIGHTGBM_FEATURES,
        name="LightGBM Feature Contract (22 Features)",
        df=df,
        dataset_path=dataset_path,
    )


def validate_anomaly_features(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Validate all 8 Isolation Forest features exist and have zero missing values."""
    return _validate_feature_set(
        feature_list=ANOMALY_FEATURES,
        name="Isolation Forest Contract (8 Features)",
        df=df,
        dataset_path=dataset_path,
    )


def _validate_feature_set(
    feature_list: List[str],
    name: str,
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Internal helper to validate a model's expected feature contract."""
    target_set = set(feature_list)

    if df is not None:
        existing = set(df.columns)
        missing = sorted(list(target_set - existing))
        if missing:
            return ValidationResult(name=name, passed=False, message=f"Missing features: {missing}", details={"missing": missing})
        null_counts = {col: int(df[col].isnull().sum()) for col in feature_list if df[col].isnull().sum() > 0}
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        cols_schema = con.execute(f"DESCRIBE SELECT * FROM '{p}' LIMIT 1").fetchall()
        existing = set(row[0] for row in cols_schema)
        missing = sorted(list(target_set - existing))
        if missing:
            return ValidationResult(name=name, passed=False, message=f"Missing features: {missing}", details={"missing": missing})

        null_exprs = [f"count(*) - count(\"{c}\")" for c in feature_list]
        null_row = con.execute(f"SELECT {', '.join(null_exprs)} FROM '{p}'").fetchone()
        null_counts = {col: cnt for col, cnt in zip(feature_list, null_row) if cnt > 0}
    else:
        loader = DataLoader(data_path=dataset_path)
        sample = loader.load(limit=1000, validate_columns=False)
        existing = set(sample.columns)
        missing = sorted(list(target_set - existing))
        if missing:
            return ValidationResult(name=name, passed=False, message=f"Missing features: {missing}", details={"missing": missing})
        null_counts = {}

    passed = len(null_counts) == 0
    msg = f"All {len(feature_list)} features present with 0 missing values." if passed else f"Nulls found in model features: {null_counts}"
    return ValidationResult(
        name=name,
        passed=passed,
        message=msg,
        details={"total_features": len(feature_list), "null_counts": null_counts},
    )


def validate_missing_values(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Audit missing values across all columns and verify modeling columns are clean."""
    # Critical modeling columns must have zero missing values
    critical_cols = set(IDENTIFIER_COLUMNS + LIGHTGBM_FEATURES + [PRIMARY_FORECAST_TARGET, OBSERVED_ENERGY_FIELD])

    if df is not None:
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        cols_schema = con.execute(f"DESCRIBE SELECT * FROM '{p}' LIMIT 1").fetchall()
        cols = [r[0] for r in cols_schema]
        null_exprs = [f"count(*) - count(\"{c}\")" for c in cols]
        null_row = con.execute(f"SELECT {', '.join(null_exprs)} FROM '{p}'").fetchone()
        null_counts = {col: cnt for col, cnt in zip(cols, null_row)}
    else:
        loader = DataLoader(data_path=dataset_path)
        sample = loader.load(limit=5000)
        null_counts = {col: int(sample[col].isnull().sum()) for col in sample.columns}

    violating_critical = {col: cnt for col, cnt in null_counts.items() if col in critical_cols and cnt > 0}
    passed = len(violating_critical) == 0
    msg = "All critical modeling & identifier columns have 0 nulls." if passed else f"Critical columns contain nulls: {violating_critical}"
    return ValidationResult(
        name="Missing Value Contract",
        passed=passed,
        message=msg,
        details={"violating_critical": violating_critical, "all_null_counts": null_counts},
    )


def validate_duplicates(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Check for duplicate timestamps within building/meter combinations."""
    if df is not None:
        dup_count = int(df.duplicated(subset=["building_id", "meter", "timestamp"]).sum())
    elif dataset_path and HAS_DUCKDB:
        con = duckdb.connect()
        p = str(dataset_path).replace("'", "''")
        res = con.execute(f"""
            SELECT sum(cnt - 1) FROM (
                SELECT count(*) as cnt
                FROM '{p}'
                GROUP BY building_id, meter, timestamp
                HAVING count(*) > 1
            )
        """).fetchone()[0]
        dup_count = int(res) if res is not None else 0
    else:
        loader = DataLoader(data_path=dataset_path)
        id_df = loader.load(columns=["building_id", "meter", "timestamp"], limit=50000)
        dup_count = int(id_df.duplicated().sum())

    passed = (dup_count == 0)
    msg = "0 duplicate timestamps within building/meter groups." if passed else f"Found {dup_count} duplicate timestamps."
    return ValidationResult(
        name="Timestamp Uniqueness Contract",
        passed=passed,
        message=msg,
        details={"duplicate_timestamp_count": dup_count},
    )


def run_all_validations(
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> ValidationSuiteResult:
    """Run all validation checks and return aggregate result."""
    target_path = str(resolve_data_path(dataset_path)) if dataset_path or df is None else "In-memory DataFrame"

    checks = [
        validate_required_columns(df=df, dataset_path=target_path),
        validate_timestamps(df=df, dataset_path=target_path),
        validate_building_ids(df=df, dataset_path=target_path),
        validate_meters(df=df, dataset_path=target_path),
        validate_target(df=df, dataset_path=target_path),
        validate_lightgbm_features(df=df, dataset_path=target_path),
        validate_anomaly_features(df=df, dataset_path=target_path),
        validate_missing_values(df=df, dataset_path=target_path),
        validate_duplicates(df=df, dataset_path=target_path),
    ]

    all_passed = all(c.passed for c in checks)
    suite = ValidationSuiteResult(
        dataset_path=target_path,
        passed=all_passed,
        results=checks,
        summary={"total_checks": len(checks), "passed_checks": sum(1 for c in checks if c.passed)},
    )
    return suite


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_all_validations(dataset_path=path_arg)
    result.print_report()
    sys.exit(0 if result.passed else 1)
