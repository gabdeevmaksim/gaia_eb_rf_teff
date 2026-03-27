#!/usr/bin/env python3
"""
Add additional EB system parameters to the unified photometry dataset.

Thin wrapper around scripts/join_columns.py:
- Target: eb_unified_photometry.parquet (default)
- Source: eb_additional_params-result.ecsv
- Join key: source_id
- Columns: global_ranking, frequency, geom_model_reference_level, model_type

Usage
-----
    python scripts/add_additional_params_to_unified.py
    python scripts/add_additional_params_to_unified.py --unified path/to/unified.parquet --params-ecsv path/to/eb_additional_params-result.ecsv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.config import get_config  # noqa: E402
from join_columns import run as join_run  # noqa: E402


def run(
    unified_path: Path,
    params_ecsv_path: Path,
    missing_value: float = -999.0,
) -> None:
    join_run(
        target_path=unified_path,
        source_path=params_ecsv_path,
        key="source_id",
        columns=["global_ranking", "frequency", "geom_model_reference_level", "model_type"],
        rename=None,
        fill_null_numeric=True,
        missing_value=missing_value,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Add EB additional parameters to eb_unified_photometry.parquet"
    )
    parser.add_argument("--unified", type=Path, default=None)
    parser.add_argument("--params-ecsv", type=Path, default=None)
    args = parser.parse_args()

    config = get_config()
    data_root = config.get_path("data_root")
    unified_path = args.unified or config.get_dataset_path("eb_unified_photometry", "raw")
    if not unified_path.is_absolute():
        unified_path = (PROJECT_ROOT / unified_path).resolve()
    params_ecsv_path = args.params_ecsv or (data_root / "raw" / "eb_additional_params-result.ecsv")
    if not params_ecsv_path.is_absolute():
        params_ecsv_path = (PROJECT_ROOT / params_ecsv_path).resolve()
    missing_value = float(config.get("processing", "missing_value", default=-999.0))

    run(
        unified_path=unified_path,
        params_ecsv_path=params_ecsv_path,
        missing_value=missing_value,
    )


if __name__ == "__main__":
    main()
