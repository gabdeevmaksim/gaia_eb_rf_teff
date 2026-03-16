#!/usr/bin/env python3
"""
Evaluate the Flag1 model on the same test sample as the base model,
including probabilistic metrics (inner uncertainty, 5-component GMM, CRPS, PIT).

Loads the Flag1 trained model directly (not catalog predictions) so that
per-tree predictions are available for GMM fitting.

The test set is reproduced from the base model config (same data, preprocessing,
and train/test split with test_size=0.2, random_state=42).

Usage:
    python scripts/evaluate_flag1_on_shared_test_probabilistic.py
    python scripts/evaluate_flag1_on_shared_test_probabilistic.py \
        --ref-config config/models/gaia_teff_corrected_log_optuna.yaml \
        --flag1-config config/models/gaia_teff_flag1_corrected_log_optuna.yaml \
        --output reports/flag1_shared_test_probabilistic.txt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture

from src.config import get_config
from src.mixture_density import (
    gaussian_mixture_crps,
    gaussian_mixture_pit,
)


def _fit_gmm_chunked(
    tree_preds_kelvin: np.ndarray,
    n_components: int,
    random_state: int,
    chunk_size: int = 10_000,
) -> tuple:
    """Fit GMM per sample in chunks to stay within memory and show progress."""
    n_trees, n_samples = tree_preds_kelvin.shape
    weights = np.empty((n_samples, n_components))
    means = np.empty((n_samples, n_components))
    sigmas = np.empty((n_samples, n_components))

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        print(f"  GMM chunk {start:,}–{end:,} / {n_samples:,}  ", end="\r", flush=True)
        for i in range(start, end):
            preds = tree_preds_kelvin[:, i].reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=random_state,
                max_iter=200,
            )
            gmm.fit(preds)
            order = np.argsort(gmm.means_.ravel())
            weights[i] = gmm.weights_[order]
            means[i] = gmm.means_.ravel()[order]
            sigmas[i] = np.sqrt(gmm.covariances_.ravel()[order])

    print()
    return weights, means, sigmas


def load_model_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_smooth_teff_correction(
    df_pd: pd.DataFrame,
    target_column: str,
    threshold: float,
    blend_width: float,
    coeffs_path: Path,
) -> pd.DataFrame:
    """Apply polynomial Teff correction with smooth Hermite blending."""
    coeffs_data = joblib.load(coeffs_path)
    if isinstance(coeffs_data, dict):
        polynomial = coeffs_data["polynomial"]
    else:
        polynomial = None

    needs_correction = (df_pd[target_column] > threshold) & (
        df_pd[target_column] != -999.0
    )
    corrected_column = f"{target_column}_corrected"
    df_pd[corrected_column] = df_pd[target_column].astype(np.float64)

    if needs_correction.any():
        teff_orig = df_pd.loc[needs_correction, target_column].values.astype(np.float64)

        if polynomial is not None:
            poly_vals = polynomial(teff_orig)
        else:
            poly_vals = np.zeros_like(teff_orig)
            for i, c in enumerate(coeffs_data):
                poly_vals += c * (teff_orig ** i)

        t = np.clip((teff_orig - threshold) / blend_width, 0.0, 1.0)
        blend = t * t * (3.0 - 2.0 * t)
        correction = (poly_vals - teff_orig) * blend
        df_pd.loc[needs_correction, corrected_column] = teff_orig + correction

    return df_pd


def reproduce_reference_test_set(
    data_path: Path,
    ref_config: dict,
    config,
    coeffs_path: Path,
) -> dict:
    """
    Reproduce the reference model's preprocessing and train/test split.

    Returns dict with keys: source_ids, features (DataFrame), teff_corrected,
    teff_raw, feature_cols.
    """
    df = pl.read_parquet(data_path)
    target = ref_config["data"]["target"]
    features = ref_config["data"]["features"]
    id_col = ref_config["data"].get("id_column", "source_id")
    preprocessing = ref_config.get("preprocessing", {})
    missing_value = preprocessing.get("missing_value", -999.0)
    filters = preprocessing.get("filters", {})
    test_size = ref_config.get("training", {}).get("test_size", 0.2)
    random_state = ref_config.get("training", {}).get("random_state", 42)

    teff_corr_cfg = ref_config.get("teff_correction", {})
    threshold = teff_corr_cfg.get("threshold", 10000)
    blend_width = teff_corr_cfg.get("blend_width", 2000)

    # Apply Teff correction (smooth blending)
    if teff_corr_cfg.get("enabled", False) and coeffs_path.exists():
        tcol = teff_corr_cfg.get("target_column", "teff_gaia")
        df_pd = df.to_pandas()
        df_pd = apply_smooth_teff_correction(df_pd, tcol, threshold, blend_width, coeffs_path)
        df = pl.from_pandas(df_pd)

    # Value filters
    for col, bounds in filters.items():
        lo, hi = bounds
        if col in df.columns:
            df = df.filter((pl.col(col) >= lo) & (pl.col(col) <= hi))

    # Drop missing
    if preprocessing.get("drop_missing", True):
        cols = [target] + [c for c in features if c in df.columns]
        df_pd = df.to_pandas()
        for c in cols:
            if c in df_pd.columns:
                df_pd[c] = df_pd[c].replace(missing_value, np.nan)
                close = np.abs(df_pd[c].astype(float) - missing_value) < 1e-6
                df_pd.loc[close, c] = np.nan
        df = pl.from_pandas(df_pd)
        df = df.drop_nulls(subset=cols)

    # Drop invalid target
    df = df.filter(pl.col(target).is_finite() & (pl.col(target) > 0))

    n = len(df)
    indices = np.arange(n)
    _, i_test = train_test_split(indices, test_size=test_size, random_state=random_state)

    df_pd = df.to_pandas()
    return {
        "source_ids": df_pd[id_col].values[i_test],
        "features": df_pd[features].iloc[i_test].reset_index(drop=True),
        "teff_corrected": df_pd["teff_gaia_corrected"].values[i_test]
            if "teff_gaia_corrected" in df_pd.columns
            else df_pd["teff_gaia"].values[i_test],
        "teff_raw": df_pd["teff_gaia"].values[i_test],
        "feature_cols": features,
        "n_total": n,
    }


def find_latest_model(models_dir: Path, pattern: str) -> Path:
    """Find most recent model matching the glob pattern (timestamp-suffixed .pkl)."""
    _ts_re = re.compile(r"_\d{8}_\d{6}$")
    candidates = sorted(
        f for f in models_dir.glob(f"{pattern}.pkl") if _ts_re.search(f.stem)
    )
    if not candidates:
        raise FileNotFoundError(f"No model found matching '{pattern}' in {models_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Flag1 model on shared test set with probabilistic metrics."
    )
    parser.add_argument(
        "--ref-config", type=Path, default=None,
        help="Reference model config defining the test set "
             "(default: config/models/gaia_teff_corrected_log_optuna.yaml)",
    )
    parser.add_argument(
        "--flag1-config", type=Path, default=None,
        help="Flag1 model config (default: config/models/gaia_teff_flag1_corrected_log_optuna.yaml)",
    )
    parser.add_argument(
        "--flag1-model", type=Path, default=None,
        help="Path to Flag1 .pkl model (default: latest matching model in models/)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save metrics summary to this file",
    )
    parser.add_argument(
        "--output-parquet", type=Path, default=None,
        help="Save per-object results (predictions, GMM, CRPS, PIT) to parquet",
    )
    args = parser.parse_args()

    config = get_config()
    project_root = config.project_root

    # --- Paths ---
    ref_config_path = args.ref_config or (
        project_root / "config" / "models" / "gaia_teff_corrected_log_optuna.yaml"
    )
    flag1_config_path = args.flag1_config or (
        project_root / "config" / "models" / "gaia_teff_flag1_corrected_log_optuna.yaml"
    )
    ref_config = load_model_config(ref_config_path)
    flag1_config = load_model_config(flag1_config_path)

    data_path = config.get_dataset_path("eb_unified_photometry", "raw")
    coeffs_file = ref_config.get("teff_correction", {}).get(
        "coefficients_file", "teff_correction_coeffs_deg2.pkl"
    )
    coeffs_path = Path(config.get_path("data_root")) / coeffs_file

    # --- Reproduce reference test set ---
    print("Reproducing reference model test set...")
    test_data = reproduce_reference_test_set(data_path, ref_config, config, coeffs_path)
    n_test = len(test_data["source_ids"])
    print(f"  Total after preprocessing: {test_data['n_total']:,}")
    print(f"  Test set size: {n_test:,}")

    # --- Load Flag1 model ---
    if args.flag1_model:
        model_path = args.flag1_model
    else:
        flag1_prefix = flag1_config["model"]["id_prefix"]
        model_path = find_latest_model(Path(config.get_path("models")), f"{flag1_prefix}_*")

    print(f"Loading Flag1 model: {model_path.name}")
    model = joblib.load(model_path)

    # Load metadata for model info
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  Model: {metadata.get('model_name', '?')}")
        print(f"  Features: {metadata.get('n_features', '?')}")

    # --- Predict ---
    X_test = test_data["features"].values.astype(np.float32)
    target_transform = flag1_config.get("target_transform", "none")

    print(f"Predicting on {n_test:,} test samples (target_transform={target_transform})...")
    y_pred_transformed = model.predict(X_test)

    # Per-tree predictions (build one at a time to allow float32 conversion)
    n_trees = len(model.estimators_)
    print(f"Collecting per-tree predictions ({n_trees} trees)...")
    tree_preds = np.empty((n_trees, n_test), dtype=np.float32)
    for t_idx, est in enumerate(model.estimators_):
        tree_preds[t_idx] = est.predict(X_test).astype(np.float32)

    # Free the model (~2-3 GB)
    del model
    import gc; gc.collect()
    print("  Model freed from memory")

    # Uncertainty in transformed space, then inverse transform
    unc_transformed = np.std(tree_preds, axis=0)

    if target_transform == "log":
        y_pred = 10 ** y_pred_transformed
        inner_uncertainty = y_pred * unc_transformed * np.log(10)
        tree_preds_kelvin = (10.0 ** tree_preds.astype(np.float64)).astype(np.float32)
    elif target_transform == "ln":
        y_pred = np.exp(y_pred_transformed)
        inner_uncertainty = y_pred * unc_transformed
        tree_preds_kelvin = np.exp(tree_preds.astype(np.float64)).astype(np.float32)
    else:
        y_pred = y_pred_transformed
        inner_uncertainty = unc_transformed.astype(np.float64)
        tree_preds_kelvin = tree_preds

    del tree_preds
    gc.collect()

    # Extract what we need and free the rest
    source_ids = test_data["source_ids"]
    y_true = test_data["teff_corrected"].astype(np.float64)
    y_true_raw = test_data["teff_raw"].astype(np.float64)
    del test_data, X_test
    gc.collect()

    # --- Fit GMM (sequential, no joblib to avoid memory duplication) ---
    n_components = 5
    random_state = flag1_config.get("training", {}).get("random_state", 42)
    print(f"Fitting {n_components}-component GMM to tree predictions "
          f"({n_trees} trees, {n_test:,} samples)...")
    gmm_weights, gmm_means, gmm_sigmas = _fit_gmm_chunked(
        tree_preds_kelvin, n_components=n_components,
        random_state=random_state,
    )
    del tree_preds_kelvin
    gc.collect()

    crps_values = gaussian_mixture_crps(gmm_weights, gmm_means, gmm_sigmas, y_true)
    pit_values = gaussian_mixture_pit(gmm_weights, gmm_means, gmm_sigmas, y_true)

    # --- Point metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rel_err = np.abs(y_pred - y_true) / y_true
    within_5 = (rel_err <= 0.05).mean() * 100
    within_10 = (rel_err <= 0.10).mean() * 100
    within_20 = (rel_err <= 0.20).mean() * 100

    # Also compute against raw teff_gaia for reference
    mae_raw = mean_absolute_error(y_true_raw, y_pred)
    rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred))
    r2_raw = r2_score(y_true_raw, y_pred)
    rel_err_raw = np.abs(y_pred - y_true_raw) / y_true_raw
    within_10_raw = (rel_err_raw <= 0.10).mean() * 100

    # --- Report ---
    lines = [
        "Flag1 model — shared test set evaluation (probabilistic)",
        "=" * 70,
        f"Reference config: {ref_config_path.name}",
        f"Flag1 model:      {model_path.name}",
        f"Test samples:     {n_test:,}",
        "",
        "POINT METRICS (vs teff_gaia_corrected):",
        f"  MAE:        {mae:.1f} K",
        f"  RMSE:       {rmse:.1f} K",
        f"  R²:         {r2:.4f}",
        f"  Within 5%:  {within_5:.1f}%",
        f"  Within 10%: {within_10:.1f}%",
        f"  Within 20%: {within_20:.1f}%",
        "",
        "POINT METRICS (vs teff_gaia, uncorrected):",
        f"  MAE:        {mae_raw:.1f} K",
        f"  RMSE:       {rmse_raw:.1f} K",
        f"  R²:         {r2_raw:.4f}",
        f"  Within 10%: {within_10_raw:.1f}%",
        "",
        "INNER UNCERTAINTY (std across trees):",
        f"  Median: {np.median(inner_uncertainty):.1f} K",
        f"  Mean:   {np.mean(inner_uncertainty):.1f} K",
        "",
        f"PROBABILISTIC METRICS ({n_components}-component GMM, vs teff_gaia_corrected):",
        f"  CRPS mean:   {np.mean(crps_values):.1f} K",
        f"  CRPS median: {np.median(crps_values):.1f} K",
        f"  PIT mean:    {np.mean(pit_values):.3f}  (ideal = 0.500)",
        f"  PIT std:     {np.std(pit_values):.3f}  (ideal ≈ 0.289)",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    # --- Save summary ---
    output_path = args.output or (
        project_root / "reports" / "flag1_shared_test_probabilistic.txt"
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved summary: {output_path}")

    # --- Save per-object parquet ---
    parquet_path = args.output_parquet or (
        project_root / "data" / "processed" / "flag1_shared_test_probabilistic.parquet"
    )
    parquet_path = Path(parquet_path).resolve()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({
        "source_id": source_ids,
        "teff_gaia": y_true_raw,
        "teff_gaia_corrected": y_true,
        "teff_predicted": y_pred,
        "y_pred_uncertainty": inner_uncertainty,
        "crps": crps_values,
        "pit": pit_values,
    })
    for k in range(n_components):
        result_df[f"gmm_weight_{k}"] = gmm_weights[:, k]
        result_df[f"gmm_mean_{k}"] = gmm_means[:, k]
        result_df[f"gmm_sigma_{k}"] = gmm_sigmas[:, k]

    result_df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet: {parquet_path}")


if __name__ == "__main__":
    main()
