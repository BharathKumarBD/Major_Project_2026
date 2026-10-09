"""Data loader for Twinergy 2.0.

Provides efficient, configurable access to the cleaned dataset (Parquet and CSV),
supporting DuckDB predicate pushdown and pandas DataFrame output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.data.schema import (
    ALL_COLUMNS,
    IDENTIFIER_COLUMNS,
    LIGHTGBM_FEATURES,
    METER_TYPE_NAMES,
)

# Optional DuckDB acceleration
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# Optional PyArrow acceleration
try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def get_default_project_root() -> Path:
    """Return the repository root directory relative to this module."""
    return Path(__file__).resolve().parents[2]


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve data file path with configurable precedence.

    Precedence:
    1. custom_path argument
    2. TWINERGY_DATA_PATH environment variable
    3. project_root/twinergy_clean.parquet (if exists)
    4. project_root/twinergy_clean.csv (if exists)
    5. project_root/data/twinergy_clean.parquet (if exists)
    6. project_root/data/twinergy_clean.csv (if exists)
    """
    if custom_path is not None:
        p = Path(custom_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {p}")
        return p

    env_path = os.getenv("TWINERGY_DATA_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (get_default_project_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"TWINERGY_DATA_PATH does not exist: {p}")
        return p

    root = get_default_project_root()
    candidates = [
        root / "twinergy_clean.parquet",
        root / "twinergy_clean.csv",
        root / "data" / "twinergy_clean.parquet",
        root / "data" / "twinergy_clean.csv",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"No dataset found in default locations under {root}. "
        "Please provide a path or set TWINERGY_DATA_PATH."
    )


class DataLoader:
    """Reusable, memory-efficient data loader for Twinergy datasets."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        use_duckdb: bool = True,
    ):
        """Initialize DataLoader.

        Args:
            data_path: Optional path to .parquet or .csv dataset.
                       If None, resolved automatically.
            use_duckdb: Whether to use DuckDB for pushdown queries if available.
        """
        self.data_path = resolve_data_path(data_path)
        self.is_parquet = self.data_path.suffix.lower() == ".parquet"
        self.use_duckdb = use_duckdb and HAS_DUCKDB

    def load(
        self,
        building_id: Optional[Union[int, Sequence[int]]] = None,
        meter: Optional[Union[int, Sequence[int]]] = None,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        validate_columns: bool = True,
    ) -> pd.DataFrame:
        """Load dataset with optional filtering by building, meter, time, and columns.

        Args:
            building_id: Single integer or sequence of building IDs to filter.
            meter: Single integer or sequence of meter IDs to filter (e.g. 0 for electricity).
            start_time: Optional start timestamp (inclusive), string or Timestamp.
            end_time: Optional end timestamp (inclusive), string or Timestamp.
            columns: Specific columns to project. If None, loads all columns.
            limit: Maximum rows to return.
            validate_columns: If True, asserts requested columns are valid.

        Returns:
            pd.DataFrame sorted by timestamp if timestamp is present.
        """
        if columns is not None and validate_columns:
            self._validate_requested_columns(columns)

        if self.use_duckdb:
            df = self._load_with_duckdb(
                building_id=building_id,
                meter=meter,
                start_time=start_time,
                end_time=end_time,
                columns=columns,
                limit=limit,
            )
        else:
            df = self._load_fallback(
                building_id=building_id,
                meter=meter,
                start_time=start_time,
                end_time=end_time,
                columns=columns,
                limit=limit,
            )

        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _validate_requested_columns(self, columns: Sequence[str]) -> None:
        """Verify that requested columns exist in the canonical schema."""
        invalid = [col for col in columns if col not in ALL_COLUMNS]
        if invalid:
            raise ValueError(f"Requested columns not recognized in schema: {invalid}")

    def _load_with_duckdb(
        self,
        building_id: Optional[Union[int, Sequence[int]]] = None,
        meter: Optional[Union[int, Sequence[int]]] = None,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query dataset using DuckDB out-of-core streaming and pushdown predicates."""
        con = duckdb.connect()

        # Build column projection
        if columns:
            select_clause = ", ".join(f'"{c}"' for c in columns)
        else:
            select_clause = "*"

        # Build WHERE clauses safely
        where_clauses: List[str] = []
        path_str = str(self.data_path).replace("'", "''")

        if building_id is not None:
            if isinstance(building_id, (int, float)):
                where_clauses.append(f"building_id = {int(building_id)}")
            else:
                b_list = ",".join(str(int(b)) for b in building_id)
                where_clauses.append(f"building_id IN ({b_list})")

        if meter is not None:
            if isinstance(meter, (int, float)):
                where_clauses.append(f"meter = {int(meter)}")
            else:
                m_list = ",".join(str(int(m)) for m in meter)
                where_clauses.append(f"meter IN ({m_list})")

        if start_time is not None:
            st_str = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
            where_clauses.append(f"timestamp >= '{st_str}'")

        if end_time is not None:
            et_str = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")
            where_clauses.append(f"timestamp <= '{et_str}'")

        query = f"SELECT {select_clause} FROM '{path_str}'"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY timestamp"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        return con.execute(query).df()

    def _load_fallback(
        self,
        building_id: Optional[Union[int, Sequence[int]]] = None,
        meter: Optional[Union[int, Sequence[int]]] = None,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fallback loader using PyArrow or pandas."""
        needed_cols: Optional[set] = set(columns) if columns else None
        cols_to_read = None
        if needed_cols is not None:
            cols_to_read = list(needed_cols)
            if building_id is not None and "building_id" not in cols_to_read:
                cols_to_read.append("building_id")
            if meter is not None and "meter" not in cols_to_read:
                cols_to_read.append("meter")
            if (start_time is not None or end_time is not None) and "timestamp" not in cols_to_read:
                cols_to_read.append("timestamp")

        if self.is_parquet:
            df = pd.read_parquet(self.data_path, columns=cols_to_read)
        else:
            df = pd.read_csv(self.data_path, usecols=cols_to_read, parse_dates=["timestamp"])

        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter in-memory
        if building_id is not None and "building_id" in df.columns:
            if isinstance(building_id, (int, float)):
                df = df[df["building_id"] == building_id]
            else:
                df = df[df["building_id"].isin(building_id)]

        if meter is not None and "meter" in df.columns:
            if isinstance(meter, (int, float)):
                df = df[df["meter"] == meter]
            else:
                df = df[df["meter"].isin(meter)]

        if start_time is not None and "timestamp" in df.columns:
            df = df[df["timestamp"] >= pd.to_datetime(start_time)]

        if end_time is not None and "timestamp" in df.columns:
            df = df[df["timestamp"] <= pd.to_datetime(end_time)]

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        if limit is not None:
            df = df.head(limit)

        if columns is not None:
            df = df[[c for c in columns if c in df.columns]]

        return df.reset_index(drop=True)

    def get_building_ids(self, meter: Optional[int] = 0) -> List[int]:
        """Return sorted list of unique building IDs (optionally filtered by meter)."""
        if self.use_duckdb:
            con = duckdb.connect()
            path_str = str(self.data_path).replace("'", "''")
            query = f"SELECT DISTINCT building_id FROM '{path_str}'"
            if meter is not None:
                query += f" WHERE meter = {int(meter)}"
            query += " ORDER BY building_id"
            res = con.execute(query).fetchall()
            return [r[0] for r in res]
        else:
            df = self.load(meter=meter, columns=["building_id"])
            return sorted(df["building_id"].unique().tolist())

    def get_meters(self) -> List[int]:
        """Return list of distinct meter IDs available in the dataset."""
        if self.use_duckdb:
            con = duckdb.connect()
            path_str = str(self.data_path).replace("'", "''")
            res = con.execute(f"SELECT DISTINCT meter FROM '{path_str}' ORDER BY meter").fetchall()
            return [r[0] for r in res]
        else:
            df = self.load(columns=["meter"])
            return sorted(df["meter"].unique().tolist())

    def get_building_metadata(self, building_id: int) -> Dict[str, Any]:
        """Return metadata dictionary for a specific building (primary_use, square_feet, site_id)."""
        cols = ["building_id", "site_id", "primary_use", "square_feet", "year_built", "floor_count"]
        df = self.load(building_id=building_id, columns=cols, limit=1)
        if len(df) == 0:
            raise ValueError(f"Building ID {building_id} not found in dataset.")
        row = df.iloc[0].to_dict()
        return row

    def get_latest_timestamp(self, building_id: int, meter: int = 0) -> pd.Timestamp:
        """Return the maximum recorded timestamp for a building and meter.

        Args:
            building_id: Target building identifier.
            meter: Target meter identifier (default 0 for electricity).

        Returns:
            pd.Timestamp of the latest recorded historical observation.
        """
        if self.use_duckdb:
            con = duckdb.connect()
            path_str = str(self.data_path).replace("'", "''")
            query = f"""
                SELECT max(timestamp)
                FROM '{path_str}'
                WHERE building_id = {int(building_id)} AND meter = {int(meter)}
            """
            res = con.execute(query).fetchone()
            if res is None or res[0] is None:
                raise ValueError(
                    f"No records found for building_id={building_id}, meter={meter} in {self.data_path}"
                )
            return pd.to_datetime(res[0])
        else:
            df = self.load(building_id=building_id, meter=meter, columns=["timestamp"])
            if len(df) == 0:
                raise ValueError(
                    f"No records found for building_id={building_id}, meter={meter} in {self.data_path}"
                )
            return pd.to_datetime(df["timestamp"].max())

    def get_monthly_summary(self, meter: int = 0) -> pd.DataFrame:
        """Compute monthly aggregate consumption (total, avg, peak) across all buildings for a meter."""
        if self.use_duckdb:
            con = duckdb.connect()
            path_str = str(self.data_path).replace("'", "''")
            query = f"""
                SELECT 
                    CAST(month AS INT) as month,
                    sum(meter_reading) as total,
                    avg(meter_reading) as avg,
                    max(meter_reading) as peak
                FROM '{path_str}'
                WHERE meter = {int(meter)}
                GROUP BY month
                ORDER BY month
            """
            return con.execute(query).df()
        else:
            df = self.load(meter=meter, columns=["month", "meter_reading"])
            monthly = df.groupby("month")["meter_reading"].agg(["sum", "mean", "max"]).reset_index()
            monthly.columns = ["month", "total", "avg", "peak"]
            return monthly


def load_data(
    building_id: Optional[Union[int, Sequence[int]]] = None,
    meter: Optional[Union[int, Sequence[int]]] = None,
    start_time: Optional[Union[str, pd.Timestamp]] = None,
    end_time: Optional[Union[str, pd.Timestamp]] = None,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Convenience function for loading dataset directly."""
    loader = DataLoader(data_path=data_path)
    return loader.load(
        building_id=building_id,
        meter=meter,
        start_time=start_time,
        end_time=end_time,
        columns=columns,
        limit=limit,
    )
