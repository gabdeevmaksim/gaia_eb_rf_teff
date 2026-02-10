#!/usr/bin/env python3
"""
Chained prediction with uncertainty propagation via Monte Carlo trajectory.

Chain: model_logg (Gaia colors → logg) → model_teff (Gaia colors + logg → log10(Teff)).

For each sample:
  1. Get predictions from ALL trees in model_logg → distribution of logg.
  2. For each logg value in that distribution: build input [colors, logg], then
     sample ONE random tree from model_teff to get teff (not the mean).
  3. Final teff mean and std over those trajectory outcomes; convert from log to Kelvin.

By default uses the latest (most recently modified) model matching each pattern:
  --logg-model  rf_gaia_logg_optuna_*.pkl
  --teff-model   rf_gaia_logg_teff_corrected_log_optuna_*.pkl
So retrained models (e.g. with restricted Optuna search space) are picked automatically.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, desc=None, total=None, **kwargs):
        return it


# -----------------------------------------------------------------------------
# Helpers: load model and data (DRY – reuse config and standard load)
# -----------------------------------------------------------------------------

def load_model_by_pattern(pattern: str, models_dir: Path):
    """
    Load the most recent model matching `pattern` (e.g. 'rf_gaia_logg_optuna_*.pkl').
    Excludes stems containing '_clustering_' or '_scaler' so only main .pkl models are used.
    Picks the latest by modification time. Returns (model, metadata_dict, model_path).
    """
    all_matches = list(models_dir.glob(pattern))
    matches = [
        p for p in all_matches
        if "_clustering_" not in p.stem and "_scaler" not in p.stem
    ]
    if not matches:
        raise FileNotFoundError(f"No model found matching: {pattern} in {models_dir}")
    model_path = max(matches, key=lambda p: p.stat().st_mtime)
    model = joblib.load(model_path)
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return model, metadata, model_path


def load_prediction_data(
    data_path: Path,
    feature_cols: list,
    id_col: str = "source_id",
    n_max: int = None,
) -> tuple:
    """
    Load parquet and return (X as ndarray, ids, feature_cols).
    ids is always returned: from id_col if present, else range(n) so output can always include source_id.
    """
    df = pd.read_parquet(data_path)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    if id_col in df.columns:
        ids = df[id_col].values
    else:
        ids = np.arange(len(df), dtype=np.int64)
    if n_max is not None and len(X) > n_max:
        X = X[:n_max]
        ids = ids[:n_max]
    return X, ids, feature_cols


# -----------------------------------------------------------------------------
# Monte Carlo trajectory: logg → teff with uncertainty propagation
# -----------------------------------------------------------------------------

def get_logg_tree_predictions(
    model_logg,
    X_colors: np.ndarray,
    tree_indices: np.ndarray = None,
    log=None,
    use_tqdm: bool = True,
) -> np.ndarray:
    """
    For each sample, get prediction from (subsampled or all) trees in model_logg.
    X_colors: (n_samples, n_features_logg), e.g. (N, 6).
    tree_indices: if set, only these tree indices are used (e.g. 100 of 900).
    Returns: (n_samples, n_trees_used).
    """
    n_samples = X_colors.shape[0]
    trees = model_logg.estimators_
    if tree_indices is not None:
        trees = [trees[i] for i in tree_indices]
    n_trees = len(trees)
    out = np.empty((n_samples, n_trees), dtype=np.float64)
    it = enumerate(trees)
    if use_tqdm and HAS_TQDM:
        it = tqdm(it, total=n_trees, desc="logg trees")
    for t, tree in it:
        out[:, t] = tree.predict(X_colors)
    if log:
        log.info(f"  Logg tree predictions: shape ({n_samples}, {n_trees})")
    return out


def propagate_uncertainty(
    model_logg,
    model_teff,
    X_colors: np.ndarray,
    features_logg: list,
    features_teff: list,
    rng: np.random.Generator,
    teff_in_log: bool = True,
    n_logg_sample: int = None,
    log=None,
    use_tqdm: bool = True,
) -> tuple:
    """
    Monte Carlo trajectory: for each sample, distribution of logg from model_logg;
    for each logg value, one random tree from model_teff → teff trajectory.
    n_logg_sample: if set, use only this many randomly chosen logg trees (faster).
    Returns (teff_mean, teff_std) in Kelvin; if teff_in_log, model_teff outputs log10(Teff).
    """
    n_samples = X_colors.shape[0]
    n_trees_logg_total = len(model_logg.estimators_)
    if n_logg_sample is not None and n_logg_sample < n_trees_logg_total:
        tree_indices = rng.choice(n_trees_logg_total, size=n_logg_sample, replace=False)
        tree_indices = np.sort(tree_indices)  # deterministic order for reproducibility given seed
    else:
        tree_indices = np.arange(n_trees_logg_total)
    n_trees_logg = len(tree_indices)
    n_trees_teff = len(model_teff.estimators_)
    if log:
        log.info(f"  Trajectories: {n_trees_logg} logg trees (of {n_trees_logg_total} total)")

    # (n_samples, n_trees_logg): each column is one tree's logg prediction
    logg_all = get_logg_tree_predictions(
        model_logg, X_colors, tree_indices=tree_indices, log=log, use_tqdm=use_tqdm
    )

    # One trajectory per logg tree: for trajectory j, use logg from tree j and one random teff tree
    teff_trajectories = np.empty((n_trees_logg, n_samples), dtype=np.float64)
    j_range = range(n_trees_logg)
    if use_tqdm and HAS_TQDM:
        j_range = tqdm(j_range, total=n_trees_logg, desc="teff trajectories")
    for j in j_range:
        logg_j = logg_all[:, j]  # (n_samples,)
        # Build teff input: [colors, logg] in feature order
        X_teff = np.hstack([X_colors, logg_j.reshape(-1, 1)])  # (n_samples, n_features_teff)
        tree_idx = rng.integers(0, n_trees_teff)
        teff_trajectories[j, :] = model_teff.estimators_[tree_idx].predict(X_teff)

    # Mean and std over trajectories (over logg-tree dimension)
    teff_mean_log = np.mean(teff_trajectories, axis=0)  # (n_samples,)
    teff_std_log = np.std(teff_trajectories, axis=0)

    if teff_in_log:
        # Convert from log10(Teff) to Teff [K]
        teff_mean_k = np.power(10.0, teff_mean_log)
        # Approx std in K: sigma_K ≈ T * sigma_log10 * ln(10)
        teff_std_k = teff_mean_k * teff_std_log * np.log(10.0)
        return teff_mean_k, teff_std_k
    else:
        return teff_mean_log, teff_std_log


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chained logg → teff prediction with Monte Carlo uncertainty propagation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input parquet (default: data/raw/eb_unified_photometry.parquet)",
    )
    parser.add_argument(
        "--logg-model",
        type=str,
        default="rf_gaia_logg_optuna_*.pkl",
        help="Glob pattern for logg model (latest match by mtime is used)",
    )
    parser.add_argument(
        "--teff-model",
        type=str,
        default="rf_gaia_logg_teff_corrected_log_optuna_*.pkl",
        help="Glob pattern for chained teff model (latest match by mtime is used)",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Max samples for test run (default: no limit, use full dataset)",
    )
    parser.add_argument(
        "--n-logg-sample",
        type=int,
        default=100,
        help="Number of logg trees to use for trajectories (default 100). Use fewer for speed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tree sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet for mean/std (default: data/processed/teff_propagated_uncertainty.parquet)",
    )
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="Write progress and summary to a .log file next to the output parquet",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args()

    config = get_config()
    models_dir = Path(config.get_path("models"))
    processed_dir = Path(config.get_path("processed"))

    out_path = args.output or (processed_dir / "teff_propagated_uncertainty.parquet")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Logging: console + optional log file
    log = logging.getLogger("propagate_uncertainty")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.addHandler(console)
    if args.log_file:
        log_path = out_path.with_suffix(".log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        log.info(f"Log file: {log_path}")

    log.info("Loading models...")
    model_logg, meta_logg, path_logg = load_model_by_pattern(args.logg_model, models_dir)
    model_teff, meta_teff, path_teff = load_model_by_pattern(args.teff_model, models_dir)
    log.info(f"  logg model: {path_logg.name}")
    log.info(f"  teff model: {path_teff.name}")

    features_logg = meta_logg.get("features", ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"])
    features_teff = meta_teff.get("features", features_logg + ["logg_gaia"])
    teff_in_log = (meta_teff.get("target_transform") or "").lower() == "log"

    data_path = args.data or config.get_dataset_path("eb_unified_photometry", "raw")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    log.info("Loading data...")
    # Load only color columns; logg is produced by model_logg in the chain. ids always returned (source_id or index).
    X_colors, ids, _ = load_prediction_data(
        data_path,
        feature_cols=features_logg,
        n_max=args.n_max,
    )
    log.info(f"  Samples: {len(X_colors):,}")

    log.info("Running Monte Carlo trajectories...")
    rng = np.random.default_rng(args.seed)
    use_tqdm = HAS_TQDM and not args.no_progress
    teff_mean, teff_std = propagate_uncertainty(
        model_logg,
        model_teff,
        X_colors,
        features_logg,
        features_teff,
        rng,
        teff_in_log=teff_in_log,
        n_logg_sample=args.n_logg_sample,
        log=log,
        use_tqdm=use_tqdm,
    )

    # Output: source_id first (by default), then teff_mean_k, teff_std_k
    result = pd.DataFrame({
        "source_id": ids,
        "teff_mean_k": teff_mean,
        "teff_std_k": teff_std,
    })
    result.to_parquet(out_path, index=False)
    log.info(f"Saved: {out_path}")
    log.info(f"  Rows: {len(result):,} | Columns: source_id, teff_mean_k, teff_std_k")
    log.info(f"  teff_mean_k: mean={teff_mean.mean():.0f} K | mean uncertainty (teff_std_k): {teff_std.mean():.0f} K")


if __name__ == "__main__":
    main()
