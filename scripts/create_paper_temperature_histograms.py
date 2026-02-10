#!/usr/bin/env python3
"""
Create publication-ready temperature histograms for paper.

Uses this repo's catalog with Teff (eb_catalog_teff.parquet).
Generates:
1. Histogram comparing Gaia-only vs ML (best) temperature distributions
2. Histogram of ML prediction uncertainties (teff_best_uncertainty)

Output: Square format, grayscale palette, saved to paper/figures/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config  # noqa: E402

# Publication settings (match other paper scripts)
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5

# Missing value for teff_gaia_original in this repo
MISSING_TEFF = -999.0


def _teff_best_to_kelvin(series: pd.Series) -> pd.Series:
    """Convert teff_best to Kelvin when it is in log10(K) (e.g. from flag1 model)."""
    out = series.copy().astype(float)
    # Heuristic: log10(Teff) is typically 3.3--4.8; Teff in K is 2000--60000
    log_mask = (series >= 3.0) & (series <= 5.0) & (series < 100)
    out.loc[log_mask] = np.power(10.0, series.loc[log_mask])
    return out


def load_data():
    """Load unified predictions; expose Gaia and ML columns for histograms."""
    config = get_config()
    processed_dir = config.get_path("processed")
    data_path = config.get_dataset_path("eb_catalog_teff", "processed")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}. Run merge script first:\n"
            "  python scripts/merge_teff_predictions_into_unified.py"
        )

    df = pd.read_parquet(data_path)
    print(f"Loading data from: {data_path}")
    print(f"  Total rows: {len(df):,}")

    # Gaia: valid original Teff (not missing)
    t = df["teff_gaia_original"].astype(float)
    valid_gaia = (
        (t != MISSING_TEFF)
        & (np.abs(t - MISSING_TEFF) > 1e-6)
        & t.notna()
        & (t > 0)
    )
    n_gaia = valid_gaia.sum()
    # ML: best prediction available (convert to K for histogram)
    has_best = df["teff_best"].notna() & np.isfinite(df["teff_best"]) & (df["teff_best"] != MISSING_TEFF)
    n_best = has_best.sum()
    teff_pred_K = _teff_best_to_kelvin(df["teff_best"])
    has_unc = df["teff_best_uncertainty"].notna() & (df["teff_best_uncertainty"] > 0)

    df = df.assign(
        teff_gaia=df["teff_gaia_original"].where(valid_gaia),
        teff_predicted=teff_pred_K.where(has_best),
        teff_uncertainty=df["teff_best_uncertainty"].where(has_unc),
    )

    print("\nCatalog statistics:")
    print(f"  Gaia Teff available: {n_gaia:,} ({100*n_gaia/len(df):.1f}%)")
    print(f"  ML (best) Teff available: {n_best:,} ({100*n_best/len(df):.1f}%)")
    print(f"  Best uncertainty available: {has_unc.sum():,}")
    return df


def create_temperature_comparison_histogram(df, output_dir):
    """Create histogram comparing Gaia-only vs ML (best) temperatures."""
    print("\nCreating temperature distribution comparison...")

    gaia_temps = df["teff_gaia"].dropna().values
    ml_temps = df["teff_predicted"].dropna().values
    if len(gaia_temps) == 0:
        print("  Skipping: no Gaia temperatures.")
        return
    if len(ml_temps) == 0:
        print("  Skipping: no ML temperatures.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    bins = np.linspace(3000, 30000, 55)

    ax.hist(
        ml_temps,
        bins=bins,
        alpha=0.7,
        color="0.7",
        edgecolor="black",
        linewidth=1.5,
        label="Predicted (best)",
    )
    ax.hist(
        gaia_temps,
        bins=bins,
        histtype="step",
        edgecolor="black",
        linewidth=2.0,
        label="GSP-Phot",
    )

    ax.set_xlabel("Effective Temperature (K)")
    ax.set_ylabel("Number of Objects")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_xlim(3000, 30000)
    ax.set_yscale("log")
    ax.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = output_dir / f"temperature_distribution_comparison.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    print(f"\nGaia: count={len(gaia_temps):,}, mean={np.mean(gaia_temps):.0f} K, median={np.median(gaia_temps):.0f} K")
    print(f"ML (best): count={len(ml_temps):,}, mean={np.mean(ml_temps):.0f} K, median={np.median(ml_temps):.0f} K")


def create_uncertainty_histogram(df, output_dir):
    """Create histogram of best-prediction uncertainties (in K when applicable)."""
    print("\nCreating uncertainty distribution histogram...")

    uncertainties = df["teff_uncertainty"].dropna().values
    uncertainties = uncertainties[np.isfinite(uncertainties) & (uncertainties > 0)]
    if len(uncertainties) == 0:
        print("  Skipping: no valid uncertainties.")
        return

    p99 = np.percentile(uncertainties, 99)
    uncertainties_filtered = uncertainties[uncertainties <= p99]

    fig, ax = plt.subplots(figsize=(8, 8))
    bins = np.linspace(0, p99, 50)
    ax.hist(
        uncertainties_filtered,
        bins=bins,
        color="0.5",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Temperature Uncertainty (K)")
    ax.set_ylabel("Number of Objects")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = output_dir / f"uncertainty_distribution.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    print(f"\nUncertainty: count={len(uncertainties):,}, mean={np.mean(uncertainties):.2g}, median={np.median(uncertainties):.2g}, 99th%={p99:.2g}")


def main():
    print("=" * 80)
    print("Creating Paper Temperature Histograms")
    print("=" * 80)

    output_dir = project_root / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    create_temperature_comparison_histogram(df, output_dir)
    create_uncertainty_histogram(df, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETED: All histograms saved to paper/figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
