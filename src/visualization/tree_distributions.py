"""
Per-object tree prediction distribution plots.

For a given test sample, shows:
  - Histogram of individual-tree predictions (the RF ensemble spread)
  - Fitted K-component Gaussian mixture (GMM) density overlay
  - Vertical line at the true value

Designed for diagnostic exploration of how the RF + GMM
predictive distributions look at different temperature ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Sequence


def plot_tree_distribution_single(
    ax,
    tree_preds: np.ndarray,
    y_true: float,
    gmm_weights: np.ndarray,
    gmm_means: np.ndarray,
    gmm_sigmas: np.ndarray,
    n_bins: int = 40,
    target_info: Optional[dict] = None,
    title: Optional[str] = None,
):
    """
    Plot the distribution of per-tree predictions for one object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    tree_preds : (n_trees,)
        Predictions from each tree for this sample.
    y_true : float
        True target value.
    gmm_weights, gmm_means, gmm_sigmas : (K,)
        GMM parameters for this sample.
    n_bins : int
        Number of histogram bins.
    target_info : dict, optional
        Keys: 'name', 'unit', 'short'.
    title : str, optional
        Panel title override.
    """
    if target_info is None:
        target_info = {"name": "Temperature", "unit": "K", "short": "Teff"}
    unit = target_info["unit"]

    ax.hist(
        tree_preds,
        bins=n_bins,
        density=True,
        color="0.75",
        edgecolor="0.50",
        linewidth=0.6,
        label="Tree predictions",
        zorder=1,
    )

    lo = min(tree_preds.min(), y_true) - 3 * np.std(tree_preds)
    hi = max(tree_preds.max(), y_true) + 3 * np.std(tree_preds)
    x_grid = np.linspace(lo, hi, 500)

    gmm_pdf = np.zeros_like(x_grid)
    for k in range(len(gmm_weights)):
        comp_pdf = (
            gmm_weights[k]
            / (gmm_sigmas[k] * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x_grid - gmm_means[k]) / gmm_sigmas[k]) ** 2)
        )
        ax.plot(x_grid, comp_pdf, "--", color="C0", alpha=0.45, linewidth=1.0, zorder=2)
        gmm_pdf += comp_pdf

    ax.plot(x_grid, gmm_pdf, "-", color="C0", linewidth=2.0, label="GMM fit", zorder=3)
    ax.axvline(y_true, color="C3", linewidth=2.0, linestyle="-", label=f"True = {y_true:.0f} {unit}", zorder=4)

    y_pred_mean = float(np.dot(gmm_weights, gmm_means))
    ax.axvline(y_pred_mean, color="C2", linewidth=1.5, linestyle="--", label=f"GMM mean = {y_pred_mean:.0f} {unit}", zorder=4)

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{target_info['short']} ({unit})", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)


def plot_tree_distributions_grid(
    tree_preds_all: Sequence[np.ndarray],
    y_true_all: Sequence[float],
    gmm_weights_all: np.ndarray,
    gmm_means_all: np.ndarray,
    gmm_sigmas_all: np.ndarray,
    bin_labels: Sequence[str],
    n_bins: int = 40,
    target_info: Optional[dict] = None,
    suptitle: Optional[str] = None,
    ncols: int = 3,
):
    """
    Grid of per-object tree-prediction distributions, one panel per Teff bin.

    Parameters
    ----------
    tree_preds_all : list of (n_trees,)
        Per-tree predictions for each selected object.
    y_true_all : list of float
        True value for each object.
    gmm_weights_all, gmm_means_all, gmm_sigmas_all : (n_objects, K)
        GMM parameters for each object.
    bin_labels : list of str
        Label for each panel (e.g. "3000–4000 K").
    n_bins : int
        Histogram bins.
    target_info : dict, optional
    suptitle : str, optional
    ncols : int
        Columns in the grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(tree_preds_all)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(n):
        ax = axes.flat[i]
        plot_tree_distribution_single(
            ax,
            tree_preds_all[i],
            y_true_all[i],
            gmm_weights_all[i],
            gmm_means_all[i],
            gmm_sigmas_all[i],
            n_bins=n_bins,
            target_info=target_info,
            title=bin_labels[i],
        )

    for j in range(n, len(axes.flat)):
        axes.flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
