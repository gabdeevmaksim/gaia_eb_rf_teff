#!/usr/bin/env python3
"""
Summarize MAE, RMSE, and R² for the test sample for all four Teff models.

Reads model metadata (clustering, log, flag1) and chain_model_metrics.txt (chain).
Writes a markdown table to paper/model_comparison_test_metrics.md and optionally CSV.

Usage:
    python scripts/summarize_model_test_metrics.py
    python scripts/summarize_model_test_metrics.py --models-dir models --output paper/model_comparison_test_metrics.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config

# Model patterns (stem prefix) -> display name (order matches merge script PREDICTION_FILES)
MODEL_PATTERNS = {
    "rf_gaia_teff_corrected_clustering_optuna": "Clustering (corrected, Optuna)",
    "rf_gaia_teff_corrected_log_optuna": "Log (corrected, Optuna)",
    "rf_gaia_teff_flag1_corrected_optuna": "Flag1 (corrected, Optuna)",
}
CHAIN_METRICS_FILE = "chain_model_metrics.txt"
CHAIN_DISPLAY_NAME = "Chain (logg → Teff)"
# Prefixes as in merge script (column names for teff and uncertainty)
PREDICTION_PREFIXES = ["teff_clustering", "teff_log", "teff_flag1", "teff_propagated"]


def get_latest_metadata(models_dir: Path, prefix: str) -> Path | None:
    """Return path to latest _metadata.json whose stem starts with prefix."""
    candidates = [p for p in models_dir.glob(f"{prefix}_*_metadata.json")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_pct_chosen_as_best(unified_path: Path) -> dict[str, dict] | None:
    """
    From unified predictions parquet, compute for each model the count and percentage
    of rows (with at least one prediction) for which that model was chosen as teff_best.
    Returns dict prefix -> {"pct": float, "count": int}, or None if file missing or no model columns.
    """
    if not unified_path.exists():
        return None
    df = pd.read_parquet(unified_path)
    teff_cols = [p for p in PREDICTION_PREFIXES if p in df.columns]
    if not teff_cols:
        return None
    unc_cols = [f"{p}_uncertainty" for p in teff_cols]
    if not all(c in df.columns for c in unc_cols):
        return None
    teff_arr = df[teff_cols].to_numpy(dtype=float)
    unc_arr = df[unc_cols].to_numpy(dtype=float)
    invalid = (
        np.isnan(teff_arr)
        | np.isnan(unc_arr)
        | (unc_arr <= 0)
        | (teff_arr == -999.0)
    )
    unc_masked = np.where(invalid, np.inf, unc_arr)
    min_unc_idx = np.nanargmin(unc_masked, axis=1)
    all_invalid = np.all(invalid, axis=1)
    n_with_best = (~all_invalid).sum()
    if n_with_best == 0:
        return {p: {"pct": 0.0, "count": 0} for p in teff_cols}
    out = {}
    for i, prefix in enumerate(teff_cols):
        wins = int(((min_unc_idx == i) & ~all_invalid).sum())
        out[prefix] = {"pct": 100.0 * wins / n_with_best, "count": wins}
    return out


def load_chain_metrics(processed_dir: Path) -> dict | None:
    """Parse chain_model_metrics.txt; return dict with mae, rmse, r2, n_samples or None."""
    path = processed_dir / CHAIN_METRICS_FILE
    if not path.exists():
        return None
    text = path.read_text()
    out = {}
    m_mae = re.search(r"MAE:\s*([\d.]+)\s*K", text)
    m_rmse = re.search(r"RMSE:\s*([\d.]+)\s*K", text)
    m_r2 = re.search(r"R²:\s*([\d.]+)", text)
    m_n = re.search(r"Test samples \(matched\):\s*([\d,]+)", text) or re.search(
        r"Test samples:\s*([\d,]+)", text
    )
    if m_mae:
        out["mae"] = float(m_mae.group(1))
    if m_rmse:
        out["rmse"] = float(m_rmse.group(1))
    if m_r2:
        out["r2"] = float(m_r2.group(1))
    if m_n:
        out["n_samples"] = int(m_n.group(1).replace(",", ""))
    if not out:
        return None
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize test metrics for all four Teff models.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Models directory (default: from config)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed data directory for chain_model_metrics.txt (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: paper/model_comparison_test_metrics.md)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write CSV to this path",
    )
    parser.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Catalog with Teff parquet for '%% chosen as best' (default: processed/eb_catalog_teff.parquet)",
    )
    args = parser.parse_args()

    config = get_config()
    models_dir = args.models_dir or config.get_path("models")
    processed_dir = args.processed_dir or config.get_path("processed")
    unified_path = args.unified or config.get_dataset_path("eb_catalog_teff", "processed")
    out_path = args.output or (PROJECT_ROOT / "paper" / "model_comparison_test_metrics.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pct_chosen = compute_pct_chosen_as_best(unified_path)

    rows = []

    # Three models from metadata
    for prefix, display_name in MODEL_PATTERNS.items():
        meta_path = get_latest_metadata(models_dir, prefix)
        if meta_path is None:
            rows.append(
                {
                    "model": display_name,
                    "mae_k": None,
                    "rmse_k": None,
                    "r2": None,
                    "n_test": None,
                }
            )
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        tm = meta.get("test_metrics") or {}
        rows.append(
            {
                "model": display_name,
                "mae_k": tm.get("mae"),
                "rmse_k": tm.get("rmse"),
                "r2": tm.get("r2"),
                "n_test": tm.get("n_samples"),
            }
        )

    # Chain from metrics file
    chain = load_chain_metrics(processed_dir)
    if chain:
        rows.append(
            {
                "model": CHAIN_DISPLAY_NAME,
                "mae_k": chain.get("mae"),
                "rmse_k": chain.get("rmse"),
                "r2": chain.get("r2"),
                "n_test": chain.get("n_samples"),
            }
        )
    else:
        rows.append(
            {
                "model": CHAIN_DISPLAY_NAME,
                "mae_k": None,
                "rmse_k": None,
                "r2": None,
                "n_test": None,
            }
        )

    # Add % chosen as best (same order as PREDICTION_PREFIXES)
    for i, prefix in enumerate(PREDICTION_PREFIXES):
        if i < len(rows):
            rows[i]["pct_chosen"] = (pct_chosen or {}).get(prefix)

    # Markdown: table 1 (test metrics) with % chosen column
    lines = [
        "# Test sample metrics: MAE, RMSE, R²",
        "",
        "Summary of test-set performance for all four Teff prediction models.",
        "",
        "| Model | MAE (K) | RMSE (K) | R² | N (test) | % chosen as best |",
        "|-------|--------|----------|-----|----------|-------------------|",
    ]
    for r in rows:
        mae = f"{r['mae_k']:.1f}" if r["mae_k"] is not None else "—"
        rmse = f"{r['rmse_k']:.1f}" if r["rmse_k"] is not None else "—"
        r2 = f"{r['r2']:.4f}" if r["r2"] is not None else "—"
        n = f"{r['n_test']:,}" if r["n_test"] is not None else "—"
        if r.get("pct_chosen") is not None:
            pct = f"{r['pct_chosen']['count']:,} ({r['pct_chosen']['pct']:.1f}%)"
        else:
            pct = "—"
        lines.append(f"| {r['model']} | {mae} | {rmse} | {r2} | {n} | {pct} |")
    lines.append("")
    lines.append("**% chosen as best:** Among objects with at least one prediction in the unified catalog, the number and fraction for which this model had the lowest uncertainty and was therefore selected as `teff_best` (format: count, percentage).")
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")

    # Optional CSV
    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["model", "mae_k", "rmse_k", "r2", "n_test", "pct_chosen", "n_chosen"],
            )
            w.writeheader()
            for r in rows:
                flat = {k: v for k, v in r.items() if k != "pct_chosen"}
                pc = r.get("pct_chosen")
                flat["pct_chosen"] = pc["pct"] if pc else None
                flat["n_chosen"] = pc["count"] if pc else None
                w.writerow(flat)
        print(f"Saved: {args.csv}")


if __name__ == "__main__":
    main()
