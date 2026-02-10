#!/usr/bin/env python3
"""
Evaluate chained (logg → Teff) prediction model on the same test set as the
Gaia+logg Teff model, comparing predicted temperatures with GSP-Phot (teff_gaia).

Reproduces the train/test split from config (test_size=0.2, random_state=42)
and the same preprocessing (filters, Teff correction, drop_missing) so that
metrics (MAE, RMSE, R²) are directly comparable with other models.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import get_config


def load_model_config(config_path: Path) -> dict:
    """Load model YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_teff_correction(
    df: pl.DataFrame,
    target_column: str,
    threshold: float,
    coeffs_path: Path,
) -> pl.DataFrame:
    """Apply polynomial Teff correction for Teff > threshold. Adds column {target}_corrected."""
    if target_column not in df.columns:
        return df
    coeffs_data = joblib.load(coeffs_path)
    df_pd = df.to_pandas()
    needs_correction = (df_pd[target_column] > threshold) & (
        df_pd[target_column] != -999.0
    )
    if not needs_correction.any():
        df_pd[f"{target_column}_corrected"] = df_pd[target_column].astype(np.float64)
        return pl.from_pandas(df_pd)
    teff_orig = df_pd.loc[needs_correction, target_column].values
    if isinstance(coeffs_data, dict):
        poly = coeffs_data["polynomial"]
        teff_corr = np.asarray(poly(teff_orig), dtype=np.float64)
    else:
        teff_corr = np.zeros_like(teff_orig, dtype=np.float64)
        for i, c in enumerate(coeffs_data):
            teff_corr += c * (teff_orig ** i)
    corrected_column = f"{target_column}_corrected"
    df_pd[corrected_column] = df_pd[target_column].astype(np.float64)
    df_pd.loc[needs_correction, corrected_column] = teff_corr
    return pl.from_pandas(df_pd)


def get_teff_model_test_set(
    data_path: Path,
    model_config: dict,
    config,
    coeffs_path: Optional[Path] = None,
) -> tuple:
    """
    Reproduce the same preprocessing and train/test split as the Gaia+logg Teff model.
    Returns (test_source_ids, test_teff_gaia) for comparison with GSP-Phot.
    """
    df = pl.read_parquet(data_path)
    target = model_config["data"]["target"]
    features = model_config["data"]["features"]
    id_col = model_config["data"].get("id_column", "source_id")
    preprocessing = model_config.get("preprocessing", {})
    missing_value = preprocessing.get("missing_value", -999.0)
    filters = preprocessing.get("filters", {})
    drop_missing = preprocessing.get("drop_missing", True)
    test_size = model_config.get("training", {}).get("test_size", 0.2)
    random_state = model_config.get("training", {}).get("random_state", 42)

    # Teff correction if target is teff_gaia_corrected and coefficients path provided
    teff_correction = model_config.get("teff_correction", {})
    if (
        teff_correction.get("enabled", False)
        and target == "teff_gaia_corrected"
        and coeffs_path is not None
        and coeffs_path.exists()
    ):
        tcol = teff_correction.get("target_column", "teff_gaia")
        thresh = teff_correction.get("threshold", 10000)
        df = apply_teff_correction(df, tcol, thresh, coeffs_path)

    # Value filters
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            df = df.filter((pl.col(col) >= lo) & (pl.col(col) <= hi))

    # Drop missing on target + features
    if drop_missing:
        cols = [target] + [c for c in features if c in df.columns]
        df_pd = df.to_pandas()
        for c in cols:
            if c in df_pd.columns:
                df_pd[c] = df_pd[c].replace(missing_value, np.nan)
                close = np.abs(df_pd[c].astype(float) - missing_value) < 1e-6
                df_pd.loc[close, c] = np.nan
        df = pl.from_pandas(df_pd)
        df = df.drop_nulls(subset=cols)

    # Valid rows: target finite and > 0 (for log-transform models)
    n_before = len(df)
    df = df.filter(
        pl.col(target).is_finite() & (pl.col(target) > 0) & ~pl.col(target).is_infinite()
    )
    if len(df) < n_before:
        n_removed = n_before - len(df)
        print(f"  Dropped {n_removed:,} rows with invalid target for log transform")

    n = len(df)
    X = df.select(features).to_pandas()
    y = df[target].to_pandas()
    ids = df[id_col].to_numpy()
    teff_gaia = df["teff_gaia"].to_numpy().astype(np.float64)

    # Same split as pipeline
    indices = np.arange(n)
    i_train, i_test = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    test_source_ids = ids[i_test]
    test_teff_gaia = teff_gaia[i_test]
    return test_source_ids, test_teff_gaia


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chained model vs GSP-Phot on same test set as Teff model."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to Teff model config (default: config/models/gaia_logg_teff_corrected_log_optuna.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to unified photometry parquet (default: config raw/eb_unified_photometry.parquet)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to chained predictions parquet (default: data/processed/teff_propagated_uncertainty.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save metrics summary text file (optional)",
    )
    args = parser.parse_args()

    config = get_config()
    project_root = config.project_root

    model_config_path = args.model_config or (
        project_root / "config" / "models" / "gaia_logg_teff_corrected_log_optuna.yaml"
    )
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = load_model_config(model_config_path)

    data_path = args.data or (
        config.get_dataset_path("eb_unified_photometry", "raw")
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    teff_corr = model_config.get("teff_correction", {})
    coeffs_path = None
    if teff_corr.get("enabled", False) and model_config["data"]["target"] == "teff_gaia_corrected":
        coeffs_path = config.get_path("data_root") / teff_corr.get(
            "coefficients_file", "teff_correction_coeffs_deg2.pkl"
        )
        if not coeffs_path.exists():
            raise FileNotFoundError(
                f"Teff correction coefficients not found: {coeffs_path}. "
                "Run: python scripts/download_datasets.py --datasets correction"
            )

    pred_path = args.predictions or (
        config.get_path("processed") / "teff_propagated_uncertainty.parquet"
    )
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    print("Reproducing test set from Teff model config...")
    test_source_ids, test_teff_gaia = get_teff_model_test_set(
        data_path, model_config, config, coeffs_path
    )
    print(f"  Test set size: {len(test_source_ids):,}")

    print("Loading chained predictions...")
    pred_df = pd.read_parquet(pred_path)
    pred_df = pred_df.rename(columns={"teff_mean_k": "teff_pred"})
    pred_df = pred_df[["source_id", "teff_pred"]]

    # Inner join: only objects that have both GSP-Phot and chained prediction
    test_df = pd.DataFrame({"source_id": test_source_ids, "teff_gaia": test_teff_gaia})
    merged = test_df.merge(pred_df, on="source_id", how="inner")
    n_eval = len(merged)
    if n_eval == 0:
        raise ValueError(
            "No overlap between test set and chained predictions. "
            "Check that teff_propagated_uncertainty.parquet was built from the same catalogue."
        )
    print(f"  Matched test samples: {n_eval:,}")

    y_true = merged["teff_gaia"].values
    y_pred = merged["teff_pred"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rel_err = np.abs(y_pred - y_true) / y_true
    within_10 = (rel_err <= 0.10).sum() / len(y_true) * 100

    lines = [
        "Chained model (logg → Teff) vs GSP-Phot (same test set as Gaia+logg Teff model)",
        "=" * 70,
        f"Test samples (matched): {n_eval:,}",
        "",
        "Metrics (predicted vs teff_gaia):",
        f"  MAE:  {mae:.1f} K",
        f"  RMSE: {rmse:.1f} K",
        f"  R²:   {r2:.4f}",
        f"  Within 10%: {within_10:.1f}%",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    if args.output:
        args.output = Path(args.output).resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
