"""
Combine probabilistic calibration plots (PIT, CRPS, GMM density)
into a single PDF file per model, with a compact metadata header.

Inputs
------
- Validation reports in ``reports/validation_report_*optuna.json``
- Corresponding PNG plots in ``reports/figures/**/`` with filenames:
    * ``{model_id}_pit_histogram.png``
    * ``{model_id}_crps_distribution.png``
    * ``{model_id}_gmm_density.png``

Outputs
-------
- One single-page PDF per model (model name + global metrics line by line, no dataset label):
    * ``{model_id}_probabilistic_diagnostics.pdf``
- For the flag1 model only, a second PDF for the shared test set:
    * ``reports/figures/flag1_shared_test_validation/flag1_shared_test_probabilistic_diagnostics.pdf``

Usage
-----
    # Combine PDFs for all models with validation reports
    python scripts/combine_probabilistic_plots.py

    # Only for a specific model_id
    python scripts/combine_probabilistic_plots.py \\
        --model-id rf_gaia_teff_corrected_log_optuna_20260304_144121
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config import get_config


_config = get_config()
REPORTS_DIR = _config.project_root / "reports"
FIGURES_ROOT = REPORTS_DIR / "figures"


def _find_plot(
    model_id: str,
    suffix: str,
) -> Optional[Path]:
    """
    Find a PNG plot for a given model_id and suffix.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. 'rf_gaia_teff_corrected_log_optuna_20260304_144121').
    suffix : str
        Plot suffix without extension (e.g. 'pit_histogram').

    Returns
    -------
    Path or None
        Path to the first matching PNG, or None if not found.
    """
    pattern = f"**/{model_id}_{suffix}.png"
    matches = list(FIGURES_ROOT.glob(pattern))
    if not matches:
        return None
    matches.sort()
    return matches[-1]


def _find_plot_by_prefix(prefix: str, suffix: str) -> Optional[Path]:
    """
    Find a PNG plot by filename prefix (e.g. 'flag1_shared_test').

    Returns
    -------
    Path or None
        Path to the first matching PNG, or None if not found.
    """
    pattern = f"**/{prefix}_{suffix}.png"
    matches = list(FIGURES_ROOT.glob(pattern))
    if not matches:
        return None
    matches.sort()
    return matches[-1]


def _load_validation_reports(
    reports_dir: Path,
    model_id_filter: Optional[str] = None,
) -> List[Path]:
    """
    Find validation report JSON files, optionally filtered by model_id.

    Parameters
    ----------
    reports_dir : Path
        Base reports directory.
    model_id_filter : str, optional
        If provided, only include reports whose JSON contains this model_id.

    Returns
    -------
    list of Path
        Paths to validation report JSON files.
    """
    all_reports = sorted(reports_dir.glob("validation_report_*optuna.json"))
    if model_id_filter is None:
        return all_reports

    filtered: List[Path] = []
    for path in all_reports:
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("model_id") == model_id_filter:
            filtered.append(path)
    return filtered


def _extract_metadata(report_path: Path) -> Tuple[str, Dict]:
    """
    Load validation metadata from a JSON report.

    Parameters
    ----------
    report_path : Path
        Path to validation report JSON.

    Returns
    -------
    model_id : str
        Model identifier.
    meta : dict
        Dictionary with model_name, validation_date, and metrics.
    """
    with report_path.open("r") as f:
        data = json.load(f)

    model_id = data.get("model_id", "unknown_model")
    meta = {
        "model_name": data.get("model_name", model_id),
        "validation_date": data.get("validation_date", ""),
        "metrics": data.get("metrics", {}),
    }
    return model_id, meta


def _load_flag1_shared_metrics(reports_dir: Path) -> Optional[Dict]:
    """
    Load shared-test metrics for the flag1 model, if available.

    Returns a dict with keys 'mae', 'rmse', 'r2', 'within_10_percent'.
    """
    txt_path = reports_dir / "flag1_shared_test_probabilistic.txt"
    if not txt_path.exists():
        return None

    mae = rmse = r2 = within_10 = None
    in_block = False

    with txt_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("POINT METRICS (vs teff_gaia_corrected"):
                in_block = True
                continue
            if in_block and not line:
                # End of this metrics block
                break
            if not in_block:
                continue

            if line.startswith("MAE:"):
                # e.g. "MAE:        653.2 K"
                try:
                    mae = float(line.split()[1])
                except Exception:
                    pass
            elif line.startswith("RMSE:"):
                try:
                    rmse = float(line.split()[1])
                except Exception:
                    pass
            elif line.startswith("R²:") or line.startswith("R^2:"):
                # e.g. "R²:         0.4298"
                try:
                    r2 = float(line.split()[1])
                except Exception:
                    pass
            elif line.startswith("Within 10%:"):
                # e.g. "Within 10%: 71.7%"
                try:
                    val = line.split()[2]  # "71.7%"
                    within_10 = float(val.rstrip("%"))
                except Exception:
                    pass

    if mae is None and rmse is None and r2 is None and within_10 is None:
        return None

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_10_percent": within_10,
    }


def _metrics_to_lines(metrics: Dict) -> List[str]:
    """Format global metrics as one line per metric (no dataset label)."""
    lines: List[str] = []
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")
    within_10 = metrics.get("within_10_percent")
    if mae is not None:
        lines.append(f"MAE: {mae:.1f} K")
    if rmse is not None:
        lines.append(f"RMSE: {rmse:.1f} K")
    if r2 is not None:
        lines.append(f"R²: {r2:.3f}")
    if within_10 is not None:
        lines.append(f"within 10%: {within_10:.1f}%")
    return lines


# DPI for PDF output (high resolution for sharp figures and text)
PDF_DPI = 300


def _write_one_pdf_page(
    pdf: PdfPages,
    model_name: str,
    metrics: Dict,
    pit_png: Optional[Path],
    crps_png: Optional[Path],
    gmm_png: Optional[Path],
    model_id_for_log: str,
) -> None:
    """
    Write a single PDF page: header (model name + metrics line by line), then
    GMM on top, PIT and CRPS below. No "Validation set" or dataset label.
    """
    fig = plt.figure(figsize=(8.27, 11.69), dpi=PDF_DPI)  # A4 portrait
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.6])

    # Header: model name (bold) and metrics (one per line), large readable font
    metric_lines = _metrics_to_lines(metrics)
    header_lines: List[str] = [model_name]
    header_lines.extend(metric_lines)
    fig.suptitle(
        "\n".join(header_lines),
        fontsize=14,
        fontweight="bold",
        y=0.98,
        ha="center",
        linespacing=1.3,
    )

    # Top: GMM density across full width
    ax_gmm = fig.add_subplot(gs[0, :])
    if gmm_png is not None:
        img = plt.imread(gmm_png)
        ax_gmm.imshow(img, interpolation="bilinear")
        ax_gmm.set_title("GMM density map", fontsize=12, fontweight="bold")
        ax_gmm.axis("off")
    else:
        ax_gmm.text(0.5, 0.5, "GMM density map missing", ha="center", va="center", fontsize=12)
        ax_gmm.axis("off")

    # Bottom left: PIT histogram
    ax_pit = fig.add_subplot(gs[1, 0])
    if pit_png is not None:
        img = plt.imread(pit_png)
        ax_pit.imshow(img, interpolation="bilinear")
        ax_pit.set_title("PIT histogram", fontsize=12, fontweight="bold")
        ax_pit.axis("off")
    else:
        ax_pit.text(0.5, 0.5, "PIT histogram missing", ha="center", va="center", fontsize=12)
        ax_pit.axis("off")

    # Bottom right: CRPS distribution
    ax_crps = fig.add_subplot(gs[1, 1])
    if crps_png is not None:
        img = plt.imread(crps_png)
        ax_crps.imshow(img, interpolation="bilinear")
        ax_crps.set_title("CRPS distribution", fontsize=12, fontweight="bold")
        ax_crps.axis("off")
    else:
        ax_crps.text(0.5, 0.5, "CRPS distribution missing", ha="center", va="center", fontsize=12)
        ax_crps.axis("off")

    plt.tight_layout(rect=(0.03, 0.02, 0.97, 0.94))
    pdf.savefig(fig, dpi=PDF_DPI)
    plt.close(fig)


def combine_plots_for_model(report_path: Path) -> None:
    """
    Combine PIT, CRPS, and GMM density plots into a PDF for one model.
    For flag1 model, creates two PDFs: one for flag1 test data, one for shared test set.
    """
    model_id, meta = _extract_metadata(report_path)
    model_name = meta.get("model_name", model_id)
    report_metrics = meta.get("metrics", {})

    pit_png = _find_plot(model_id, "pit_histogram")
    crps_png = _find_plot(model_id, "crps_distribution")
    gmm_png = _find_plot(model_id, "gmm_density")

    if not any(p is not None for p in (pit_png, crps_png, gmm_png)):
        print(f"[WARN] No probabilistic plots found for model_id={model_id}. Skipping.")
        return

    first_plot_dir = next(p.parent for p in (pit_png, crps_png, gmm_png) if p is not None)

    # PDF 1: model's own validation/test plots and metrics
    output_pdf = first_plot_dir / f"{model_id}_probabilistic_diagnostics.pdf"
    print(f"\n=== {model_id} ===")
    print(f"Writing: {output_pdf}")
    with PdfPages(output_pdf) as pdf:
        _write_one_pdf_page(
            pdf, model_name, report_metrics, pit_png, crps_png, gmm_png, model_id
        )

    # PDF 2 (flag1 only): shared test set plots and metrics
    if "flag1" in model_id:
        shared_metrics = _load_flag1_shared_metrics(REPORTS_DIR)
        if shared_metrics is not None:
            pit_shared = _find_plot_by_prefix("flag1_shared_test", "pit_histogram")
            crps_shared = _find_plot_by_prefix("flag1_shared_test", "crps_distribution")
            gmm_shared = _find_plot_by_prefix("flag1_shared_test", "gmm_density")
            if any(p is not None for p in (pit_shared, crps_shared, gmm_shared)):
                shared_dir = FIGURES_ROOT / "flag1_shared_test_validation"
                shared_dir.mkdir(parents=True, exist_ok=True)
                output_shared = shared_dir / "flag1_shared_test_probabilistic_diagnostics.pdf"
                print(f"Writing (flag1 shared test): {output_shared}")
                with PdfPages(output_shared) as pdf:
                    _write_one_pdf_page(
                        pdf,
                        model_name,
                        shared_metrics,
                        pit_shared,
                        crps_shared,
                        gmm_shared,
                        "flag1_shared_test",
                    )
            else:
                print("[WARN] No flag1_shared_test plots found; skipping second PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine PIT, CRPS, and GMM density plots into a PDF per model."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="If provided, only create a PDF for this model_id.",
    )
    args = parser.parse_args()

    reports = _load_validation_reports(REPORTS_DIR, model_id_filter=args.model_id)
    if not reports:
        if args.model_id is not None:
            print(f"No validation reports found for model_id={args.model_id}")
        else:
            print("No validation reports found in 'reports/validation_report_*optuna.json'.")
        return

    for report_path in reports:
        print(f"\nProcessing validation report: {report_path}")
        try:
            combine_plots_for_model(report_path)
        except Exception as exc:
            print(f"[ERROR] Failed to combine plots for {report_path}: {exc}")


if __name__ == "__main__":
    main()

