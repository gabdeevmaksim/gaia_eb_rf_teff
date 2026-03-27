#!/usr/bin/env python3
"""
Universal script to add columns to a dataset by joining another file on a key column.

By default the target (accepting) file is the unified photometry dataset.
Supports Parquet, CSV, and ECSV for both target and source.
Join key can be any column; default is source_id.

Usage
-----
    # Add columns from a CSV to the unified dataset (default target), join on source_id
    python scripts/join_columns.py --source data/raw/ebv_results.csv --columns ebv_sandf,ebv_sfd

    # Specify target and source explicitly
    python scripts/join_columns.py \\
        --target data/raw/eb_unified_photometry.parquet \\
        --source data/raw/eb_additional_params-result.ecsv \\
        --key source_id

    # Rename columns when adding (source_name:target_name)
    python scripts/join_columns.py --source data/raw/ebv_results.csv \\
        --columns ebv_sandf,ebv_sfd --rename ebv_sfd:ebv_sdf

    # Different key column names in target vs source
    python scripts/join_columns.py --source other.csv --key-target id --key-source source_id

    # Restrict columns added from source (default: all except key)
    python scripts/join_columns.py --source data/raw/params.parquet --columns col1,col2,col3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402


def _load_table(path: Path) -> pl.DataFrame:
    """Load a table from Parquet, CSV, or ECSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pl.read_parquet(path)
    if suf == ".csv":
        return pl.read_csv(path)
    if suf == ".ecsv":
        from astropy.table import Table

        table = Table.read(path, format="ascii.ecsv")
        return pl.from_pandas(table.to_pandas())
    raise ValueError(f"Unsupported format: {suf}. Use .parquet, .csv, or .ecsv.")


def _write_table(df: pl.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet or CSV based on extension."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df.write_parquet(path)
    elif suf == ".csv":
        df.write_csv(path)
    else:
        raise ValueError(f"Unsupported output format: {suf}. Use .parquet or .csv.")


def run(
    target_path: Path,
    source_path: Path,
    key: str | tuple[str, str] = "source_id",
    columns: Optional[list[str]] = None,
    rename: Optional[dict[str, str]] = None,
    fill_null_numeric: bool = True,
    missing_value: float = -999.0,
) -> None:
    """
    Add columns from source into target by left-joining on a key column.

    Parameters
    ----------
    target_path : Path
        File to add columns to (accepting dataset). Updated in-place.
    source_path : Path
        File to join from (provides new columns).
    key : str or (str, str)
        Join key. If str, same column name in both tables.
        If (key_target, key_source), different names in target vs source.
    columns : list of str, optional
        Columns from source to add. If None, add all columns except the key.
    rename : dict, optional
        Rename source columns when adding, e.g. {"ebv_sfd": "ebv_sdf"}.
    fill_null_numeric : bool
        If True, fill nulls in added numeric columns with missing_value.
    missing_value : float
        Value used to fill nulls in numeric added columns.
    """
    rename = rename or {}

    if isinstance(key, (list, tuple)):
        key_target, key_source = key[0], key[1]
    else:
        key_target = key_source = key

    print(f"Loading target (accepting) file: {target_path}")
    target = _load_table(target_path)
    print(f"  Rows: {target.height:,}, columns: {len(target.columns)}")

    print(f"Loading source file: {source_path}")
    source = _load_table(source_path)
    print(f"  Rows: {source.height:,}, columns: {len(source.columns)}")

    if key_source not in source.columns:
        raise ValueError(f"Key column '{key_source}' not found in source. Available: {list(source.columns)}")
    if key_target not in target.columns and key_target != key_source:
        raise ValueError(f"Key column '{key_target}' not found in target. Available: {list(target.columns)}")

    # Columns to take from source (excluding key)
    all_source_cols = [c for c in source.columns if c != key_source]
    if columns is not None:
        missing = set(columns) - set(source.columns)
        if missing:
            raise ValueError(f"Columns not in source: {sorted(missing)}")
        add_cols = [c for c in columns if c in source.columns]
    else:
        add_cols = all_source_cols

    if not add_cols:
        print("No columns to add from source. Exiting.")
        return

    # Build source slice: key (possibly renamed to match target) + columns to add (possibly renamed)
    source_key = key_source
    source_join = source.select([source_key] + add_cols)

    # Rename key in source if target uses a different name
    if key_target != key_source:
        source_join = source_join.rename({key_source: key_target})
        join_on = key_target
    else:
        join_on = key_target

    # Apply renames for added columns
    for old_name, new_name in rename.items():
        if old_name in source_join.columns and old_name != new_name:
            source_join = source_join.rename({old_name: new_name})

    # Avoid duplicate column names: drop from source any that already exist in target
    to_drop = [c for c in source_join.columns if c != join_on and c in target.columns]
    if to_drop:
        source_join = source_join.drop(to_drop)
        add_cols = [c for c in source_join.columns if c != join_on]

    print(f"Joining on '{join_on}' (left join), adding columns: {[c for c in source_join.columns if c != join_on]}")
    result = target.join(source_join, on=join_on, how="left")

    if fill_null_numeric:
        numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        for col in result.columns:
            if col in source_join.columns and col != join_on:
                if result[col].dtype in numeric_dtypes:
                    result = result.with_columns(pl.col(col).fill_null(missing_value))

    _write_table(result, target_path)
    added = [c for c in result.columns if c not in target.columns]
    print(f"Saved updated target to: {target_path}")
    print(f"  Rows: {result.height:,}, columns: {len(result.columns)}")
    print(f"  Added columns: {added}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add columns to a dataset by joining another file on a key column."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target (accepting) file to add columns to (default: config eb_unified_photometry in raw)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source file to join from (parquet, csv, or ecsv)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="source_id",
        help="Join key column name, same in both files (default: source_id)",
    )
    parser.add_argument(
        "--key-target",
        type=str,
        default=None,
        help="Join key column name in target (use with --key-source if names differ)",
    )
    parser.add_argument(
        "--key-source",
        type=str,
        default=None,
        help="Join key column name in source (use with --key-target if names differ)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated list of columns from source to add (default: all except key)",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default=None,
        help="Comma-separated renames when adding, e.g. ebv_sfd:ebv_sdf",
    )
    parser.add_argument(
        "--no-fill-null",
        action="store_true",
        help="Do not fill nulls in added numeric columns with missing_value",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=None,
        help="Value for filling nulls in numeric added columns (default: from config)",
    )

    args = parser.parse_args()

    config = get_config()
    data_root = config.get_path("data_root")

    target_path = args.target or config.get_dataset_path("eb_unified_photometry", "raw")
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / target_path).resolve()

    source_path = args.source
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if args.key_target is not None and args.key_source is not None:
        key = (args.key_target, args.key_source)
    else:
        key = args.key

    columns = None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    rename = {}
    if args.rename:
        for part in args.rename.split(","):
            part = part.strip()
            if ":" in part:
                old, new = part.split(":", 1)
                rename[old.strip()] = new.strip()

    missing_value = args.missing_value
    if missing_value is None:
        missing_value = float(config.get("processing", "missing_value", default=-999.0))

    run(
        target_path=target_path,
        source_path=source_path,
        key=key,
        columns=columns,
        rename=rename if rename else None,
        fill_null_numeric=not args.no_fill_null,
        missing_value=missing_value,
    )


if __name__ == "__main__":
    main()
