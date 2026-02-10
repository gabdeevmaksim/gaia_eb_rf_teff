"""
Create paper-quality validation plots for the manuscript.

Generates 4 plots without titles, square aspect ratio:
1. Feature importance
2. True vs Predicted
3. RMSE by temperature range
4. Percentage within 10%

Paths: models dir and default output dir come from config (config.get_path).
Usage:
    python scripts/create_paper_validation_plots.py
    python scripts/create_paper_validation_plots.py --model-id rf_gaia_teff_flag1_corrected_optuna_20260130_131419
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_config

# Resolve from config so models and output dir are consistent with project layout
_config = get_config()
MODEL_DIR = _config.get_path("models")
PAPER_DIR = _config.project_root / "paper" / "figures"


def get_latest_model_id(pattern: str, model_dir: Path = None) -> str:
    """Return the most recent model ID matching pattern (e.g. rf_gaia_teff_flag1_corrected_optuna_*)."""
    model_dir = model_dir or MODEL_DIR
    matches = sorted(model_dir.glob(pattern + "*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No model found matching: {pattern}* in {model_dir}")
    return matches[-1].stem


def main():
    parser = argparse.ArgumentParser(description="Create paper-quality validation plots.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID (e.g. rf_gaia_teff_flag1_corrected_optuna_20260130_131419). "
             "Default: latest rf_gaia_teff_flag1_corrected_optuna_*",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: paper/figures)",
    )
    args = parser.parse_args()

    if args.model_id is not None:
        model_id = args.model_id
    else:
        model_id = get_latest_model_id("rf_gaia_teff_flag1_corrected_optuna_")

    paper_dir = Path(args.output_dir) if args.output_dir is not None else PAPER_DIR
    paper_dir.mkdir(parents=True, exist_ok=True)

    run_plots(model_id=model_id, model_dir=MODEL_DIR, paper_dir=paper_dir)


def run_plots(model_id: str, model_dir: Path, paper_dir: Path) -> None:

    # Load model metadata
    metadata_file = model_dir / f"{model_id}_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load test predictions
    test_pred_file = model_dir / f"{model_id}_test_predictions.parquet"
    if not test_pred_file.exists():
        raise FileNotFoundError(f"Test predictions not found: {test_pred_file}")
    test_pred = pd.read_parquet(test_pred_file)

    # Rename columns for consistency and add residual
    if 'y_true' in test_pred.columns and 'y_pred' in test_pred.columns:
        test_pred = test_pred.rename(columns={'y_true': 'true_value', 'y_pred': 'predicted_value'})
    test_pred['residual'] = test_pred['predicted_value'] - test_pred['true_value']

    # Extract metrics
    mae = metadata['test_metrics']['mae']
    rmse = metadata['test_metrics']['rmse']
    r2 = metadata['test_metrics']['r2']
    within_10 = metadata['test_metrics']['within_10_percent']

    # Extract feature importance
    features = metadata['features']
    summary_file = model_dir / f"{model_id}_SUMMARY.txt"
    feature_importance = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'TOP 10 FEATURES:' in line:
                for j in range(1, min(len(features) + 1, 11)):
                    if i + j < len(lines):
                        parts = lines[i + j].split()
                        if len(parts) >= 3:
                            feature_name = parts[1]
                            importance_val = float(parts[2])
                            feature_importance[feature_name] = importance_val
    else:
        # Fallback: build from metadata feature_importances list
        imp_list = metadata.get('feature_importances', [])
        for i, f in enumerate(features):
            feature_importance[f] = imp_list[i] if i < len(imp_list) else 0.0

    print(f"Loaded model: {model_id}")
    print(f"Test predictions: {len(test_pred)} samples")
    print(f"Feature importance: {len(feature_importance)} features")

    # Calculate bin statistics for temperature ranges
    temp_bins = [0, 4000, 5000, 6000, 8000, 50000]
    temp_labels = ['<4000', '4000-5000', '5000-6000', '6000-8000', '>8000']
    test_pred['temp_bin'] = pd.cut(test_pred['true_value'], bins=temp_bins, labels=temp_labels)
    test_pred['abs_error'] = np.abs(test_pred['residual'])

    bin_stats = []
    for bin_label in temp_labels:
        mask = test_pred['temp_bin'] == bin_label
        subset = test_pred[mask]

        if len(subset) > 0:
            mae_bin = subset['abs_error'].mean()
            rmse_bin = np.sqrt((subset['residual']**2).mean())
            mean_pct = (100 * subset['abs_error'] / subset['true_value']).mean()
            within_10_bin = ((100 * subset['abs_error'] / subset['true_value']) <= 10).sum()
            within_10_pct = 100 * within_10_bin / len(subset)

            bin_stats.append({
                'bin': bin_label,
                'count': len(subset),
                'mae': mae_bin,
                'rmse': rmse_bin,
                'mean_pct': mean_pct,
                'within_10': within_10_pct
            })

    bin_stats_df = pd.DataFrame(bin_stats)

    print("\nCreating paper-quality plots...")

    # Set publication style - grayscale for print
    plt.style.use('default')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    # ============================================================================
    # PLOT 1: Feature Importance (Square, No Title)
    # ============================================================================
    print("\n1. Feature importance...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract and sort
    features_list = list(feature_importance.keys())
    importances = list(feature_importance.values())

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [features_list[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    # Plot - grayscale
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importances, color='0.5', edgecolor='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(paper_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'feature_importance.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'feature_importance.png'}")
    print(f"   Saved: {paper_dir / 'feature_importance.pdf'}")
    plt.close()

    # ============================================================================
    # PLOT 2: True vs Predicted (Square, No Title)
    # ============================================================================
    print("\n2. True vs Predicted...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Hexbin plot for density - grayscale
    hb = ax.hexbin(test_pred['true_value'], test_pred['predicted_value'],
                   gridsize=50, cmap='Greys', mincnt=1, bins='log')

    # 1:1 line
    data_min = min(test_pred['true_value'].min(), test_pred['predicted_value'].min())
    data_max = max(test_pred['true_value'].max(), test_pred['predicted_value'].max())
    ax.plot([data_min, data_max], [data_min, data_max], 'k--', lw=2.5, label='1:1')

    # ±10% lines
    x = np.array([data_min, data_max])
    ax.plot(x, x * 1.1, 'k:', lw=1.5, alpha=0.7, label='±10%')
    ax.plot(x, x * 0.9, 'k:', lw=1.5, alpha=0.7)

    ax.set_xlabel(r'$T_{\rm eff,\,true}$ (K)', fontsize=14)
    ax.set_ylabel(r'$T_{\rm eff,\,predicted}$ (K)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)

    # Remove left (y-axis) tick labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(left=False)

    # Colorbar without label
    cbar = plt.colorbar(hb, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(paper_dir / 'true_vs_predicted.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'true_vs_predicted.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'true_vs_predicted.png'}")
    print(f"   Saved: {paper_dir / 'true_vs_predicted.pdf'}")
    plt.close()

    # ============================================================================
    # PLOT 3a: RMSE by Temperature Range - Bar Chart (Square, No Title, Horizontal Labels)
    # ============================================================================
    print("\n3a. RMSE by temperature range (bar chart)...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Grayscale bars
    ax.bar(bin_stats_df['bin'], bin_stats_df['rmse'], color='0.5', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('RMSE (K)', fontsize=14)
    ax.set_xlabel(r'$T_{\rm eff}$ (K)', fontsize=14)
    ax.tick_params(axis='x', rotation=0)  # Horizontal labels
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(paper_dir / 'rmse_by_temperature_bar.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'rmse_by_temperature_bar.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'rmse_by_temperature_bar.png'}")
    print(f"   Saved: {paper_dir / 'rmse_by_temperature_bar.pdf'}")
    plt.close()

    # ============================================================================
    # PLOT 3b: RMSE by Temperature Range - Line Plot with 100K bins
    # ============================================================================
    print("\n3b. RMSE by temperature range (line plot, 100K bins)...")

    # Create finer bins with 100K steps
    teff_min = int(test_pred['true_value'].min() / 100) * 100
    teff_max = int(test_pred['true_value'].max() / 100) * 100 + 100
    temp_bins_fine = np.arange(teff_min, teff_max + 100, 100)

    # Calculate RMSE for each bin
    bin_centers = []
    rmse_values = []

    for i in range(len(temp_bins_fine) - 1):
        bin_start = temp_bins_fine[i]
        bin_end = temp_bins_fine[i + 1]

        mask = (test_pred['true_value'] >= bin_start) & (test_pred['true_value'] < bin_end)
        subset = test_pred[mask]

        if len(subset) >= 10:  # Only include bins with at least 10 samples
            bin_center = (bin_start + bin_end) / 2
            rmse_bin = np.sqrt((subset['residual']**2).mean())

            bin_centers.append(bin_center)
            rmse_values.append(rmse_bin)

    # Create line plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(bin_centers, rmse_values, linestyle='none', marker='o', markersize=4,
            markerfacecolor='0.5', markeredgecolor='black', markeredgewidth=1)
    ax.set_ylabel('RMSE (K)', fontsize=14)
    ax.set_xlabel(r'$T_{\rm eff}$ (K)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Format x-axis to show major ticks every 2500K, range (2600, 21500)
    from matplotlib.ticker import MultipleLocator
    ax.set_xlim(2600, 17500)
    ax.xaxis.set_major_locator(MultipleLocator(2500))

    plt.tight_layout()
    fig.savefig(paper_dir / 'rmse_by_temperature_line.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'rmse_by_temperature_line.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'rmse_by_temperature_line.png'}")
    print(f"   Saved: {paper_dir / 'rmse_by_temperature_line.pdf'}")
    plt.close()

    # ============================================================================
    # PLOT 3c: Absolute error vs T_eff - all points (no binning)
    # ============================================================================
    print("\n3c. Absolute error vs T_eff (all points, no binning)...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # One dot per sample: x = true Teff, y = |residual| (K)
    ax.scatter(
        test_pred['true_value'],
        test_pred['abs_error'],
        s=1,
        c='0.4',
        alpha=0.4,
        edgecolors='none',
    )
    ax.set_ylabel(r'$|T_{\rm eff,\,pred} - T_{\rm eff,\,true}|$ (K)', fontsize=14)
    ax.set_xlabel(r'$T_{\rm eff,\,true}$ (K)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2600, 18000)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(2500))

    plt.tight_layout()
    fig.savefig(paper_dir / 'abs_error_vs_teff_all_points.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'abs_error_vs_teff_all_points.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'abs_error_vs_teff_all_points.png'}")
    print(f"   Saved: {paper_dir / 'abs_error_vs_teff_all_points.pdf'}")
    plt.close()

    # ============================================================================
    # PLOT 4: Percentage Within 10% (Square, No Title)
    # ============================================================================
    print("\n4. Percentage within 10%...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Grayscale bars
    ax.bar(bin_stats_df['bin'], bin_stats_df['within_10'], color='0.5', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Objects Within 10% (%)', fontsize=14)
    ax.set_xlabel(r'$T_{\rm eff}$ (K)', fontsize=14)
    ax.tick_params(axis='x', rotation=0)  # Horizontal labels
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis ticks every 10%
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(10))

    plt.tight_layout()
    fig.savefig(paper_dir / 'percentage_within_10pct.png', dpi=300, bbox_inches='tight')
    fig.savefig(paper_dir / 'percentage_within_10pct.pdf', bbox_inches='tight')
    print(f"   Saved: {paper_dir / 'percentage_within_10pct.png'}")
    print(f"   Saved: {paper_dir / 'percentage_within_10pct.pdf'}")
    plt.close()

    print("\n" + "="*80)
    print("PAPER PLOTS CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {paper_dir.absolute()}")
    print(f"\nModel: {metadata.get('model_name', model_id)}")
    print(f"MAE: {mae:.1f} K")
    print(f"RMSE: {rmse:.1f} K")
    print(f"R²: {r2:.4f}")
    print(f"Within 10%: {within_10:.1f}%")
    print("\nAll plots saved in both PNG (300 DPI) and PDF formats.")


if __name__ == "__main__":
    main()
