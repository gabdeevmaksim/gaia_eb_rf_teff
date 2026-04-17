"""
Gaussian Mixture Density utilities for Random Forest predictions.

Fits a K-component Gaussian mixture to per-tree RF predictions for each
sample, then provides CRPS and PIT metrics following the formulation in
src/dcmdn/LossFunctions.py (Polsterer & D'Isanto, 2017).

All functions operate on plain numpy arrays -- no Theano dependency.
"""

import math
import os
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.special import erf
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Fit GMM to tree predictions (parallelized)
# ---------------------------------------------------------------------------

def _fit_single_gmm(
    preds: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one GMM to a 1-D array of tree predictions for a single sample."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=200,
    )
    gmm.fit(preds.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel())
    return (
        gmm.weights_[order],
        gmm.means_.ravel()[order],
        np.sqrt(gmm.covariances_.ravel()[order]),
    )


def _fit_single_gmm_bic(
    preds: np.ndarray,
    k_max: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit GMMs with K = 1..k_max and pick the K that minimizes BIC.

    Returns arrays padded to length k_max (unused slots filled with
    weight=0, mean=0, sigma=1) so the output stays rectangular.
    """
    X = preds.reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_k = 1

    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    order = np.argsort(best_gmm.means_.ravel())
    w = best_gmm.weights_[order]
    m = best_gmm.means_.ravel()[order]
    s = np.sqrt(best_gmm.covariances_.ravel()[order])

    # Pad to k_max so all samples have the same array width
    w_pad = np.zeros(k_max)
    m_pad = np.zeros(k_max)
    s_pad = np.ones(k_max)  # sigma=1 for unused slots (avoids div-by-zero)
    w_pad[:best_k] = w
    m_pad[:best_k] = m
    s_pad[:best_k] = s

    return w_pad, m_pad, s_pad, best_k


def fit_gmm_to_tree_predictions_bic(
    tree_predictions: np.ndarray,
    k_max: int = 8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Gaussian mixture per sample, selecting K via BIC.

    For each sample, fits K = 1..k_max and keeps the model with the
    lowest BIC.  Output arrays are padded to k_max columns; unused
    components have weight 0.

    Parameters
    ----------
    tree_predictions : (n_trees, n_samples)
    k_max : int
        Maximum number of components to try (default 8).
    random_state : int
    n_jobs : int

    Returns
    -------
    weights : (n_samples, k_max)
    means   : (n_samples, k_max)
    sigmas  : (n_samples, k_max)
    best_ks : (n_samples,)  -- selected K per sample
    """
    n_trees, n_samples = tree_predictions.shape

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_single_gmm_bic)(
            tree_predictions[:, i], k_max, random_state
        )
        for i in range(n_samples)
    )

    weights = np.array([r[0] for r in results])
    means = np.array([r[1] for r in results])
    sigmas = np.array([r[2] for r in results])
    best_ks = np.array([r[3] for r in results])

    return weights, means, sigmas, best_ks


def fit_gmm_to_tree_predictions(
    tree_predictions: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Gaussian mixture to per-tree predictions for every sample.

    Parameters
    ----------
    tree_predictions : np.ndarray, shape (n_trees, n_samples)
        Each row is one tree's prediction for all samples.
    n_components : int
        Number of Gaussian components (default 5).
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    weights : np.ndarray, shape (n_samples, n_components)
        Mixture weights (sum to 1 per sample).
    means : np.ndarray, shape (n_samples, n_components)
        Component means.
    sigmas : np.ndarray, shape (n_samples, n_components)
        Component standard deviations (>0).
    """
    n_trees, n_samples = tree_predictions.shape

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_single_gmm)(
            tree_predictions[:, i], n_components, random_state
        )
        for i in range(n_samples)
    )

    weights = np.array([r[0] for r in results])
    means = np.array([r[1] for r in results])
    sigmas = np.array([r[2] for r in results])

    return weights, means, sigmas


# ---------------------------------------------------------------------------
# Gaussian helpers (numpy equivalents of LossFunctions.py)
# ---------------------------------------------------------------------------

def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * np.square(x))


def _Phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _A(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """CRPS kernel for a single Gaussian component (Eq. from Grimit et al.)."""
    sd = np.sqrt(var)
    z = mu / sd
    return 2.0 * sd * _phi(z) + mu * (2.0 * _Phi(z) - 1.0)


# ---------------------------------------------------------------------------
# CRPS  (Continuous Ranked Probability Score)
# ---------------------------------------------------------------------------

def gaussian_mixture_crps(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    CRPS for a Gaussian mixture, per sample.

    Closed-form expression following Grimit et al. (2006) and
    src/dcmdn/LossFunctions.GaussianMixtureCRPS.

    Parameters
    ----------
    weights : (n_samples, K)
    means   : (n_samples, K)
    sigmas  : (n_samples, K)
    y_true  : (n_samples,)

    Returns
    -------
    crps : (n_samples,)  -- lower is better
    """
    K = weights.shape[1]
    y = np.asarray(y_true).ravel()
    var = np.square(sigmas)

    crps = np.zeros(len(y))
    for k in range(K):
        crps += weights[:, k] * _A(y - means[:, k], var[:, k])

    for m in range(K):
        for n in range(K):
            crps -= 0.5 * weights[:, m] * weights[:, n] * _A(
                means[:, m] - means[:, n],
                var[:, m] + var[:, n],
            )

    return crps


# ---------------------------------------------------------------------------
# PIT  (Probability Integral Transform)
# ---------------------------------------------------------------------------

def gaussian_mixture_pit(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    PIT value = CDF of the predictive mixture evaluated at the true value.

    Parameters
    ----------
    weights : (n_samples, K)
    means   : (n_samples, K)
    sigmas  : (n_samples, K)
    y_true  : (n_samples,)

    Returns
    -------
    pit : (n_samples,)  -- should be Uniform(0,1) if well-calibrated
    """
    K = weights.shape[1]
    y = np.asarray(y_true).ravel()
    pit = np.zeros(len(y))
    for k in range(K):
        z = (y - means[:, k]) / sigmas[:, k]
        pit += weights[:, k] * _Phi(z)
    return pit


# ---------------------------------------------------------------------------
# Summary statistics from a GMM prediction
# ---------------------------------------------------------------------------

def mixture_point_estimate(
    weights: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    """Mixture mean (weighted average of component means)."""
    return np.sum(weights * means, axis=1)


def mixture_variance(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Total variance of the mixture (law of total variance)."""
    mixture_mean = mixture_point_estimate(weights, means)
    var_within = np.sum(weights * np.square(sigmas), axis=1)
    var_between = np.sum(weights * np.square(means - mixture_mean[:, None]), axis=1)
    return var_within + var_between
