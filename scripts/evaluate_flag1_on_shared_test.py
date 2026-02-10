#!/usr/bin/env python3
"""
Evaluate the Flag1 model on the same test sample as the other models (clustering, log, chain).

Uses the test set defined by gaia_teff_corrected_log_optuna (or another reference config):
same data, preprocessing, and train/test split (test_size=0.2, random_state=42).
Requires predictions for objects WITH original Teff, i.e. run prediction with
config/prediction/predict_gaia_original_teff_flag1_corrected_optuna.yaml first.

Usage:
    python scripts/evaluate_flag1_on_shared_test.py
    python scripts/evaluate_flag1_on_shared_test.py --model-config config/models/gaia_teff_corrected_log_optuna.yaml --predictions data/processed/predictions_gaia_original_teff_flag1_corrected_optuna.parquet --output data/processed/flag1_metrics_shared_test.txt
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_config

# Reuse test-set logic from evaluate_chained_model
_spec = importlib.util.spec_from_file_location(
    "evaluate_chained_model",
    PROJECT_ROOT / "scripts" / "evaluate_chained_model.py",
)
_eval_chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_chain)
get_teff_model_test_set = _eval_chain.get_teff_model_test_set
load_model_config = _eval_chain.load_model_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Flag1 model on same test set as clustering/log/chain (255k samples)."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Reference model config for test set (default: config/models/gaia_teff_corrected_log_optuna.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Unified photometry parquet (default: config raw/eb_unified_photometry.parquet)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Flag1 predictions for objects with original Teff (default: processed/predictions_gaia_original_teff_flag1_corrected_optuna.parquet)",
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
        project_root / "config" / "models" / "gaia_teff_corrected_log_optuna.yaml"
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
        coeffs_path = project_root / "data" / teff_corr.get(
            "coefficients_file", "teff_correction_coeffs_deg2.pkl"
        )
        if not coeffs_path.exists():
            coeffs_path = config.get_path("data_root") / teff_corr.get(
                "coefficients_file", "teff_correction_coeffs_deg2.pkl"
            )
        if not coeffs_path.exists():
            raise FileNotFoundError(
                f"Teff correction coefficients not found: {coeffs_path}. "
                "Run: python scripts/download_datasets.py --datasets correction"
            )

    pred_path = args.predictions or (
        config.get_path("processed") / "predictions_gaia_original_teff_flag1_corrected_optuna.parquet"
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}. "
            "Run prediction with config/prediction/predict_gaia_original_teff_flag1_corrected_optuna.yaml first."
        )

    print("Reproducing test set from reference model config...")
    test_source_ids, test_teff_gaia = get_teff_model_test_set(
        data_path, model_config, config, coeffs_path
    )
    print(f"  Test set size: {len(test_source_ids):,}")

    print("Loading Flag1 predictions (objects with original Teff)...")
    pred_df = pd.read_parquet(pred_path)
    # Column may be teff_predicted or teff_pred
    pred_col = "teff_predicted" if "teff_predicted" in pred_df.columns else "teff_pred"
    pred_df = pred_df[["source_id", pred_col]].rename(columns={pred_col: "teff_pred"})

    test_df = pd.DataFrame({"source_id": test_source_ids, "teff_gaia": test_teff_gaia})
    merged = test_df.merge(pred_df, on="source_id", how="inner")
    n_eval = len(merged)
    if n_eval == 0:
        raise ValueError(
            "No overlap between test set and Flag1 predictions. "
            "Ensure predictions_gaia_original_teff_flag1_corrected_optuna.parquet was built from the same catalogue."
        )
    print(f"  Matched test samples: {n_eval:,}")

    y_true = merged["teff_gaia"].values
    y_pred = merged["teff_pred"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = (mean_squared_error(y_true, y_pred)) ** 0.5
    r2 = r2_score(y_true, y_pred)
    rel_err = np.abs(y_pred - y_true) / y_true
    within_10 = (rel_err <= 0.10).sum() / len(y_true) * 100

    lines = [
        "Flag1 model vs GSP-Phot (same test set as clustering / log / chain)",
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
