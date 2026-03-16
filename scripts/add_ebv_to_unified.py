#!/usr/bin/env python3
"""
Add E(B-V) extinction columns to the unified photometry dataset.

This script:
- Loads `eb_unified_photometry.parquet` from the raw data directory
- Loads `ebv_results.csv` (containing extinction values) from `data/raw`
- Left-joins on `source_id`
- Adds two columns to the unified dataset:
  - `ebv_sandf` (from `ebv_sandf` in the CSV)
  - `ebv_sdf`   (from `ebv_sfd` in the CSV; note the name mapping)
- Fills missing extinction values with the configured `missing_value` (default: -999.0)
- Overwrites the original unified parquet file in-place

Usage
-----
    cd /path/to/project
    python scripts/add_ebv_to_unified.py

    # Optional: custom paths
    python scripts/add_ebv_to_unified.py \\
        --unified data/raw/eb_unified_photometry.parquet \\
        --ebv-csv data/raw/ebv_results.csv
"""

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402


def run(
    unified_path: Path,
    ebv_csv_path: Path,
    missing_value: float = -999.0,
) -> None:
    """
    Add extinction columns to the unified photometry parquet in-place.
    """
    if not unified_path.exists():
        raise FileNotFoundError(f"Unified photometry file not found: {unified_path}")

    if not ebv_csv_path.exists():
        raise FileNotFoundError(f"EBV CSV file not found: {ebv_csv_path}")

    print(f"Loading unified photometry: {unified_path}")
    unified = pl.read_parquet(unified_path)
    print(f"  Unified rows: {unified.height:,}, columns: {len(unified.columns)}")

    print(f"Loading EBV results: {ebv_csv_path}")
    ebv = pl.read_csv(ebv_csv_path)

    required_cols = {"source_id", "ebv_sandf", "ebv_sfd"}
    missing = required_cols - set(ebv.columns)
    if missing:
        raise ValueError(f"EBV CSV is missing required columns: {sorted(missing)}")

    # Select and rename EBV columns:
    # - Keep ebv_sandf as-is
    # - Map ebv_sfd (file) → ebv_sdf (unified dataset) as requested
    ebv_small = ebv.select(
        [
            "source_id",
            "ebv_sandf",
            "ebv_sfd",
        ]
    ).rename(
        {
            "ebv_sfd": "ebv_sdf",
        }
    )

    print("Joining EBV columns into unified dataset on source_id...")
    unified_with_ebv = unified.join(
        ebv_small,
        on="source_id",
        how="left",
    )

    # Fill missing EBV values with configured missing_value
    for col in ("ebv_sandf", "ebv_sdf"):
        if col in unified_with_ebv.columns:
            unified_with_ebv = unified_with_ebv.with_columns(
                pl.col(col).fill_null(missing_value)
            )

    # Persist back to the same parquet path
    print("Writing updated unified dataset (with EBV) back to disk...")
    unified_with_ebv.write_parquet(unified_path)

    print(f"Saved updated unified photometry to: {unified_path}")
    print(f"  Rows: {unified_with_ebv.height:,}")
    print(f"  Columns: {len(unified_with_ebv.columns)}")
    print("  Added columns: ebv_sandf, ebv_sdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add EBV extinction columns to eb_unified_photometry.parquet"
    )
    parser.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Unified photometry parquet (default from config: eb_unified_photometry in raw)",
    )
    parser.add_argument(
        "--ebv-csv",
        type=Path,
        default=None,
        help="EBV CSV file (default: data/raw/ebv_results.csv)",
    )

    args = parser.parse_args()

    config = get_config()
    data_root = config.get_path("data_root")

    unified_path = args.unified or config.get_dataset_path("eb_unified_photometry", "raw")
    if not unified_path.is_absolute():
        unified_path = (PROJECT_ROOT / unified_path).resolve()

    ebv_csv_path = args.ebv_csv or (data_root / "raw" / "ebv_results.csv")
    if not ebv_csv_path.is_absolute():
        ebv_csv_path = (PROJECT_ROOT / ebv_csv_path).resolve()

    missing_value = float(config.get("processing", "missing_value", default=-999.0))

    run(
        unified_path=unified_path,
        ebv_csv_path=ebv_csv_path,
        missing_value=missing_value,
    )


if __name__ == "__main__":
    main()

