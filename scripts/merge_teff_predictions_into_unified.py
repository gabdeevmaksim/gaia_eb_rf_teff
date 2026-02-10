#!/usr/bin/env python3
"""
Merge Teff predictions from four model outputs into the eclipsing binary catalog.

- Input: eb_unified_photometry.parquet from data/raw (source_id + teff_gaia → teff_gaia_original).
- Applies polynomial correction to GSP-Phot Teff > 10000 K using teff_correction_coeffs_deg2.pkl,
  adding teff_gaia_corrected (same as original for Teff <= 10000 K or missing).
- Left-joins the four prediction parquets on source_id with model-coded column names.
- Adds teff_best and teff_best_uncertainty (lowest-uncertainty prediction per row).
- Output: eb_catalog_teff.parquet in data/processed (eclipsing binary catalog with Teff).

Prediction files → columns:
  - predictions_gaia_teff_corrected_clustering_optuna.parquet → teff_clustering, teff_clustering_uncertainty
  - predictions_gaia_teff_corrected_log_optuna.parquet       → teff_log, teff_log_uncertainty
  - predictions_gaia_teff_flag1_corrected_optuna.parquet     → teff_flag1, teff_flag1_uncertainty
  - teff_propagated_uncertainty.parquet (chain)              → teff_propagated, teff_propagated_uncertainty

Usage:
    python scripts/merge_teff_predictions_into_unified.py
    python scripts/merge_teff_predictions_into_unified.py --output data/processed/eb_catalog_teff.parquet
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config

# Teff correction: threshold (K) and coefficients filename
TEFF_CORRECTION_THRESHOLD = 10000
TEFF_CORRECTION_COEFFS_FILE = "teff_correction_coeffs_deg2.pkl"

# Prediction file paths (relative to processed dir) and their output column prefixes.
# Prefix "teff_propagated" = chain model: only predictions for objects that have another model's prediction (847k) are joined.
PREDICTION_FILES = [
    ("predictions_gaia_teff_corrected_clustering_optuna.parquet", "teff_clustering", "teff_gaia_corrected_predicted", "teff_gaia_corrected_uncertainty"),
    ("predictions_gaia_teff_corrected_log_optuna.parquet", "teff_log", "teff_gaia_corrected_predicted", "teff_gaia_corrected_uncertainty"),
    ("predictions_gaia_teff_flag1_corrected_optuna.parquet", "teff_flag1", "teff_predicted", "teff_uncertainty"),
    ("teff_propagated_uncertainty.parquet", "teff_propagated", "teff_mean_k", "teff_std_k"),
]


def apply_teff_correction_to_series(
    series: pd.Series,
    coeffs_path: Path,
    threshold: float,
    missing_value: float,
) -> pd.Series:
    """Apply polynomial correction for values > threshold; return corrected series (same index)."""
    out = series.astype(float).copy()
    valid = (
        series.notna()
        & (series != missing_value)
        & (np.abs(series.astype(float) - missing_value) > 1e-6)
        & (series > threshold)
    )
    if not valid.any():
        return out
    coeffs_data = joblib.load(coeffs_path)
    teff_orig = series.loc[valid].values
    if isinstance(coeffs_data, dict) and "polynomial" in coeffs_data:
        poly = coeffs_data["polynomial"]
        teff_corr = np.asarray(poly(teff_orig), dtype=np.float64)
    else:
        coeffs_array = np.atleast_1d(coeffs_data)
        teff_corr = np.zeros_like(teff_orig, dtype=np.float64)
        for i, c in enumerate(coeffs_array):
            teff_corr += c * (teff_orig ** i)
    out.loc[valid] = teff_corr
    return out


def run(
    unified_path: Path,
    processed_dir: Path,
    output_path: Path,
    missing_value: float = -999.0,
    coeffs_path: Path | None = None,
) -> None:
    """Merge unified with all prediction files and compute teff_best."""
    print("Loading unified dataset...")
    df = pd.read_parquet(unified_path, columns=["source_id", "teff_gaia"])
    df = df.rename(columns={"teff_gaia": "teff_gaia_original"})
    n_unified = len(df)
    print(f"  Rows: {n_unified:,}")

    # Apply GSP-Phot Teff correction for Teff > 10000 K
    if coeffs_path is not None and coeffs_path.exists():
        print(f"Applying Teff correction (Teff > {TEFF_CORRECTION_THRESHOLD} K)...")
        df["teff_gaia_corrected"] = apply_teff_correction_to_series(
            df["teff_gaia_original"],
            coeffs_path,
            threshold=TEFF_CORRECTION_THRESHOLD,
            missing_value=missing_value,
        )
        n_corrected = (
            (df["teff_gaia_original"] > TEFF_CORRECTION_THRESHOLD)
            & (df["teff_gaia_original"] != missing_value)
        ).sum()
        print(f"  Corrected {n_corrected:,} rows (Teff > {TEFF_CORRECTION_THRESHOLD} K)")
    else:
        df["teff_gaia_corrected"] = df["teff_gaia_original"].astype(float)
        if coeffs_path is not None:
            print(f"  Warning: coefficients not found at {coeffs_path}, using uncorrected Teff")

    # Objects without original Teff (same subset as clustering/log/flag1 predictions)
    no_teff = (
        (df["teff_gaia_original"] == missing_value)
        | (np.abs(df["teff_gaia_original"].astype(float) - missing_value) < 1e-6)
        | pd.isna(df["teff_gaia_original"])
    )
    no_teff_ids = set(df.loc[no_teff, "source_id"])
    print(f"  Objects without original Teff: {len(no_teff_ids):,}")

    prefixes = []
    for filename, prefix, teff_col, unc_col in PREDICTION_FILES:
        path = processed_dir / filename
        if not path.exists():
            print(f"  Skipping (not found): {filename}")
            continue
        print(f"Loading {filename}...")
        pred = pd.read_parquet(path, columns=["source_id", teff_col, unc_col])
        pred = pred.rename(columns={teff_col: prefix, unc_col: f"{prefix}_uncertainty"})
        # Chain model: join only for objects that have a prediction from another model (847,486),
        # i.e. same subset as clustering/log/flag1 (no original Teff and no missing features).
        if prefix == "teff_propagated":
            other_cols = [p for p in prefixes if p in df.columns]
            if other_cols:
                ids_with_other_pred = set(df.loc[df[other_cols].notna().any(axis=1), "source_id"])
                pred = pred[pred["source_id"].isin(ids_with_other_pred)]
                print(f"  Rows joined (only objects with other-model predictions): {len(pred):,}")
            else:
                pred = pred[pred["source_id"].isin(no_teff_ids)]
                print(f"  Rows joined (only objects without original Teff): {len(pred):,}")
        else:
            print(f"  Rows in file: {len(pred):,}")
        df = df.merge(pred, on="source_id", how="left")
        prefixes.append(prefix)

    if not prefixes:
        raise FileNotFoundError("No prediction files found.")

    print("Computing teff_best and teff_best_uncertainty (lowest uncertainty among predictions)...")
    teff_cols = [p for p in prefixes]
    unc_cols = [f"{p}_uncertainty" for p in prefixes]
    # Vectorized: stack teff and uncertainty, then take teff and uncertainty where uncertainty is min per row
    teff_arr = df[teff_cols].to_numpy(dtype=float)
    unc_arr = df[unc_cols].to_numpy(dtype=float)
    # Mask invalid: NaN or non-positive uncertainty or teff == -999
    invalid = np.isnan(teff_arr) | np.isnan(unc_arr) | (unc_arr <= 0) | (teff_arr == -999.0)
    unc_masked = np.where(invalid, np.inf, unc_arr)
    min_unc_idx = np.nanargmin(unc_masked, axis=1)
    # For rows where all are invalid, nanargmin returns 0; check if that row had all inf
    all_invalid = np.all(invalid, axis=1)
    best_teff = np.take_along_axis(teff_arr, min_unc_idx[:, np.newaxis], axis=1).squeeze(axis=1)
    best_teff = np.where(all_invalid, np.nan, best_teff)
    best_unc = np.take_along_axis(unc_arr, min_unc_idx[:, np.newaxis], axis=1).squeeze(axis=1)
    best_unc = np.where(all_invalid, np.nan, best_unc)
    df["teff_best"] = best_teff
    df["teff_best_uncertainty"] = best_unc

    # Reorder columns: source_id, teff_gaia_original, teff_gaia_corrected, then model columns, then teff_best, teff_best_uncertainty
    model_cols = []
    for p in prefixes:
        model_cols.append(p)
        model_cols.append(f"{p}_uncertainty")
    out_cols = ["source_id", "teff_gaia_original", "teff_gaia_corrected"] + model_cols + ["teff_best", "teff_best_uncertainty"]
    df = df[out_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    n_with_best = df["teff_best"].notna().sum()
    print(f"  Rows with teff_best: {n_with_best:,}")


def main():
    parser = argparse.ArgumentParser(description="Merge Teff predictions into unified dataset.")
    parser.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Input photometry parquet (default: config raw/eb_unified_photometry.parquet)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing prediction parquets (default: config processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: data/processed/eb_catalog_teff.parquet)",
    )
    args = parser.parse_args()

    config = get_config()
    processed_dir = args.processed_dir or config.get_path("processed")
    data_root = config.get_path("data_root")
    unified_path = args.unified or config.get_dataset_path("eb_unified_photometry", "raw")
    output_path = args.output or config.get_dataset_path("eb_catalog_teff", "processed")
    missing_value = float(config.get("processing", "missing_value", default=-999.0))
    coeffs_path = data_root / TEFF_CORRECTION_COEFFS_FILE

    run(
        unified_path=unified_path,
        processed_dir=processed_dir,
        output_path=output_path,
        missing_value=missing_value,
        coeffs_path=coeffs_path,
    )


if __name__ == "__main__":
    main()
