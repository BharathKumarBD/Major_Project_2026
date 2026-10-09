"""Data preparation script for Twinergy 2.0.

Converts raw/cleaned CSV datasets into optimized columnar Parquet format
and creates sample development datasets for fast local development and testing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pyarrow.parquet as pq

from src.data.loader import get_default_project_root


def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    compression: str = "ZSTD",
) -> None:
    """Convert cleaned CSV dataset into columnar Parquet format using DuckDB."""
    print(f"Reading CSV from : {csv_path}")
    print(f"Writing Parquet to: {parquet_path} (Compression: {compression})")
    
    t0 = time.time()
    csv_str = str(csv_path).replace("'", "''")
    parq_str = str(parquet_path).replace("'", "''")
    
    con = duckdb.connect()
    query = f"""
        COPY (SELECT * FROM '{csv_str}')
        TO '{parq_str}'
        (FORMAT PARQUET, COMPRESSION {compression})
    """
    con.execute(query)
    elapsed = time.time() - t0
    
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    parq_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    ratio = csv_size_mb / parq_size_mb if parq_size_mb > 0 else 0
    
    print(f"[SUCCESS] Conversion complete in {elapsed:.2f}s!")
    print(f"   CSV Size    : {csv_size_mb:.1f} MB")
    print(f"   Parquet Size: {parq_size_mb:.1f} MB (Compression ratio: {ratio:.1f}x)")


def create_sample_dataset(
    source_path: Path,
    sample_path: Path,
    sample_buildings: list[int] | None = None,
) -> None:
    """Create a lightweight development sample dataset for quick testing and local dev."""
    if sample_buildings is None:
        sample_buildings = [0, 1, 2, 20, 100, 500]
    
    print(f"Creating sample dataset for buildings {sample_buildings}...")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    b_str = ", ".join(str(b) for b in sample_buildings)
    src_str = str(source_path).replace("'", "''")
    dst_str = str(sample_path).replace("'", "''")
    
    con = duckdb.connect()
    query = f"""
        COPY (
            SELECT * FROM '{src_str}'
            WHERE building_id IN ({b_str})
            ORDER BY building_id, meter, timestamp
        )
        TO '{dst_str}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    size_mb = sample_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Sample dataset created at: {sample_path} ({size_mb:.2f} MB)")


def main():
    root = get_default_project_root()
    parser = argparse.ArgumentParser(description="Twinergy 2.0 Data Preparation Pipeline")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "twinergy_clean.csv",
        help="Path to source twinergy_clean.csv",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=root / "twinergy_clean.parquet",
        help="Output path for twinergy_clean.parquet",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        default=True,
        help="Create development sample parquet dataset under data/sample/",
    )
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"❌ Source CSV not found: {args.csv}")
        sys.exit(1)
        
    convert_csv_to_parquet(args.csv, args.parquet)
    
    if args.create_sample:
        sample_path = root / "data" / "sample" / "twinergy_sample.parquet"
        create_sample_dataset(args.parquet, sample_path)


if __name__ == "__main__":
    main()
