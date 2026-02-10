"""
Create a paper-quality histogram of uncertainties for the best Teff prediction.

Reads eb_catalog_teff.parquet and plots the distribution of
teff_best_uncertainty (uncertainty of the chosen best prediction per object).
Units can be Kelvin or log10(K) depending on which model was selected.

Usage:
    python scripts/plot_teff_best_uncertainty_histogram.py
    python scripts/plot_teff_best_uncertainty_histogram.py --input data/processed/eb_catalog_teff.parquet --output-dir paper/figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Histogram of best Teff uncertainties.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Catalog with Teff parquet (default: data/processed/eb_catalog_teff.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: paper/figures)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins (default: 80)",
    )
    args = parser.parse_args()

    config = get_config()
    processed_dir = config.get_path("processed")
    paper_dir = args.output_dir or (PROJECT_ROOT / "paper" / "figures")
    paper_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or config.get_dataset_path("eb_catalog_teff", "processed")
    if not input_path.exists():
        raise FileNotFoundError(f"Catalog not found: {input_path}")

    df = pd.read_parquet(input_path, columns=["teff_best_uncertainty"])
    u = df["teff_best_uncertainty"].dropna()
    u = u[(u > 0) & np.isfinite(u)]
    n = len(u)
    if n == 0:
        raise ValueError("No valid teff_best_uncertainty values in the dataset.")

    # Paper style (match create_paper_validation_plots.py)
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13

    fig, ax = plt.subplots(figsize=(8, 8))

    # Histogram: use log scale on x if span is large (mixed K vs log10 units)
    u_min, u_max = u.min(), u.max()
    if u_max / max(u_min, 1e-30) > 1e3:
        # Mixed units (e.g. 0.01 log10 and 200 K) -> log-scale x
        bins = np.logspace(np.log10(max(u_min, 1e-6)), np.log10(u_max * 1.01), args.bins)
        ax.hist(u, bins=bins, color="0.5", edgecolor="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(
            r"Uncertainty of best $T_{\rm eff}$ (K)",
            fontsize=14,
        )
    else:
        ax.hist(u, bins=args.bins, color="0.5", edgecolor="black", linewidth=0.8)
        ax.set_xlabel(
            r"Uncertainty of best $T_{\rm eff}$ (K)",
            fontsize=14,
        )

    ax.set_ylabel("Number of objects", fontsize=14)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    p50 = u.median()
    p90 = u.quantile(0.90)
    ax.axvline(p50, color="k", linestyle="--", linewidth=2, label=f"Median (50%) = {p50:.3g}")
    ax.axvline(p90, color="k", linestyle="-", linewidth=2, label=f"90% of objects = {p90:.3g}")
    ax.legend(fontsize=12)

    plt.tight_layout()
    base = "teff_best_uncertainty_histogram"
    fig.savefig(paper_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(paper_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {paper_dir / f'{base}.png'}")
    print(f"Saved: {paper_dir / f'{base}.pdf'}")
    print(f"  Objects with valid uncertainty: {n:,}")
    print(f"  Median (50%) uncertainty: {p50:.4g}")
    print(f"  90% of objects below:     {p90:.4g}")
    print(f"  Mean uncertainty:         {u.mean():.4g}")


if __name__ == "__main__":
    main()
