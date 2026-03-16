#!/usr/bin/env python3
"""
Generate validation plots for the Flag1 model evaluated on the shared test set.

Reads the parquet produced by evaluate_flag1_on_shared_test_probabilistic.py
and generates the same set of plots as the pipeline validation step, saved to
a separate directory so existing validation outputs are not overwritten.

Usage:
    python scripts/plot_flag1_shared_test_validation.py
    python scripts/plot_flag1_shared_test_validation.py \
        --input data/processed/flag1_shared_test_probabilistic.parquet \
        --outdir reports/figures/flag1_shared_test_validation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_config
from src.visualization import validation_plots


TARGET_INFO = {"name": "Temperature", "unit": "K", "short": "Teff"}
MODEL_ID = "flag1_shared_test"


def compute_bin_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE / RMSE / mean % error / within-10% per temperature bin."""
    bins = [0, 4000, 5000, 6000, 7000, 8000, 10_000, 15_000, 50_000]
    labels = [
        "<4k", "4–5k", "5–6k", "6–7k", "7–8k", "8–10k", "10–15k", ">15k"
    ]
    df = df.copy()
    df["bin"] = pd.cut(df["true_value"], bins=bins, labels=labels)

    rows = []
    for label in labels:
        subset = df[df["bin"] == label]
        if len(subset) == 0:
            continue
        y_t = subset["true_value"].values
        y_p = subset["predicted_value"].values
        rel = np.abs(y_p - y_t) / y_t
        rows.append({
            "bin": label,
            "mae": mean_absolute_error(y_t, y_p),
            "rmse": np.sqrt(mean_squared_error(y_t, y_p)),
            "mean_pct": np.mean(rel) * 100,
            "within_10": (rel <= 0.10).mean() * 100,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation plots for Flag1 shared-test results."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Parquet with per-object results (default: data/processed/flag1_shared_test_probabilistic.parquet)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output subdirectory under reports/figures/ (default: flag1_shared_test_validation)",
    )
    args = parser.parse_args()

    config = get_config()
    project_root = config.project_root

    input_path = args.input or (
        project_root / "data" / "processed" / "flag1_shared_test_probabilistic.parquet"
    )
    subdir = args.outdir or "flag1_shared_test_validation"

    print(f"Reading {input_path.name}...")
    raw = pd.read_parquet(input_path)
    n = len(raw)
    print(f"  {n:,} objects, {len(raw.columns)} columns")

    # Rename to match validation_plots conventions
    df = raw.rename(columns={
        "teff_gaia_corrected": "true_value",
        "teff_predicted": "predicted_value",
    })
    df["residual"] = df["predicted_value"] - df["true_value"]
    df["pct_error"] = np.abs(df["residual"]) / df["true_value"] * 100

    # Global metrics
    y_true = df["true_value"].values
    y_pred = df["predicted_value"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # --- 1. Test scatter ---
    print("\n--- Generating plots ---")
    validation_plots.plot_test_scatter(
        test_pred=df, mae=mae, rmse=rmse, r2=r2,
        model_id=MODEL_ID, subdir=subdir,
        model_name="Flag1 (shared test)", target_info=TARGET_INFO,
    )

    # --- 2. Residuals ---
    validation_plots.plot_residuals(
        test_pred=df, model_id=MODEL_ID, subdir=subdir,
        target_info=TARGET_INFO,
    )

    # --- 3. Performance by temperature bin ---
    bin_stats = compute_bin_stats(df)
    validation_plots.plot_performance_by_temp(
        test_pred=df, bin_stats_df=bin_stats,
        model_id=MODEL_ID, subdir=subdir,
        target_info=TARGET_INFO,
    )

    # --- 4. Temperature distributions (predicted only — no separate train data) ---
    # Build a minimal "train" DataFrame with the true values for comparison
    train_proxy = pd.DataFrame({"teff_gspphot": df["true_value"]})
    pred_proxy = pd.DataFrame({"teff_predicted": df["predicted_value"]})
    validation_plots.plot_temp_distributions(
        train_data=train_proxy, predictions=pred_proxy,
        model_id=MODEL_ID, subdir=subdir,
        train_col="teff_gspphot", pred_col="teff_predicted",
        model_name="Flag1 (shared test)", target_info=TARGET_INFO,
    )

    # --- 5. PIT histogram ---
    if "pit" in df.columns:
        validation_plots.plot_pit_histogram(
            pit_values=df["pit"].values,
            model_id=MODEL_ID, subdir=subdir,
            target_info=TARGET_INFO,
        )

    # --- 6. CRPS distribution ---
    if "crps" in df.columns:
        validation_plots.plot_crps_distribution(
            crps_values=df["crps"].values,
            model_id=MODEL_ID, subdir=subdir,
            target_info=TARGET_INFO,
        )

    # --- 7. GMM density map ---
    has_gmm = "gmm_weight_0" in df.columns
    if has_gmm:
        validation_plots.plot_gmm_density_map(
            test_pred=df, model_id=MODEL_ID, subdir=subdir,
            target_info=TARGET_INFO,
        )

    print(f"\nAll plots saved to: reports/figures/{subdir}/")


if __name__ == "__main__":
    main()
