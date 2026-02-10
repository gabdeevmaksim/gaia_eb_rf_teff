"""
Feature engineering utilities for temperature prediction models.

This module contains reusable feature engineering functions that can be
used in both notebooks and pipeline scripts.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression


def create_polynomial_features(
    df: pd.DataFrame,
    color_cols: List[str],
    degree: int = 3,
    interaction_only: bool = False
) -> pd.DataFrame:
    """
    Create polynomial features from color indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with color columns
    color_cols : list of str
        Column names to create polynomials from
    degree : int, default 3
        Maximum polynomial degree
    interaction_only : bool, default False
        If True, only interaction features (no x^2, x^3)

    Returns
    -------
    pd.DataFrame
        Dataframe with original and polynomial features

    Examples
    --------
    >>> color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
    >>> df_poly = create_polynomial_features(df, color_cols, degree=3)
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )

    # Get polynomial features
    poly_features = poly.fit_transform(df[color_cols])

    # Create column names
    poly_names = poly.get_feature_names_out(color_cols)

    # Create dataframe with polynomial features
    df_poly = df.copy()
    for i, name in enumerate(poly_names):
        if name not in color_cols:  # Don't duplicate original features
            # Clean up feature names
            clean_name = name.replace(' ', '_')
            df_poly[clean_name] = poly_features[:, i]

    return df_poly


def create_interaction_features(
    df: pd.DataFrame,
    color_cols: List[str]
) -> pd.DataFrame:
    """
    Create pairwise interaction features (color1 * color2).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with color columns
    color_cols : list of str
        Column names to create interactions from

    Returns
    -------
    pd.DataFrame
        Dataframe with original and interaction features

    Examples
    --------
    >>> df_int = create_interaction_features(df, ['g_r_color', 'r_i_color', 'bp_rp'])
    """
    df_int = df.copy()

    for i, col1 in enumerate(color_cols):
        for col2 in color_cols[i+1:]:
            interaction_name = f"{col1}_x_{col2}"
            df_int[interaction_name] = df[col1] * df[col2]

    return df_int


def create_log_features(
    df: pd.DataFrame,
    color_cols: List[str],
    offset: float = 0.5
) -> pd.DataFrame:
    """
    Create logarithmic features from colors.

    Adds an offset to handle negative colors and replaces NaN with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with color columns
    color_cols : list of str
        Column names to create log features from
    offset : float, default 0.5
        Offset to add before taking log (to handle negative values)

    Returns
    -------
    pd.DataFrame
        Dataframe with original and log features

    Examples
    --------
    >>> df_log = create_log_features(df, ['g_r_color', 'r_i_color'])
    """
    df_log = df.copy()

    for col in color_cols:
        log_name = f"log_{col}"
        # Add offset and take log, then replace any NaN with 0
        log_values = np.log(df[col] + offset)
        df_log[log_name] = log_values.fillna(0)

    return df_log


def create_temperature_dependent_features(
    df: pd.DataFrame,
    color_cols: List[str],
    hot_threshold: float = 0.3,
    cool_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Create temperature-regime-specific features.

    Creates indicator features for hot/cool/mid temperature regimes
    based on color indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with color columns
    color_cols : list of str
        Column names to use
    hot_threshold : float, default 0.3
        Color threshold for hot stars
    cool_threshold : float, default 0.8
        Color threshold for cool stars

    Returns
    -------
    pd.DataFrame
        Dataframe with temperature-dependent features

    Examples
    --------
    >>> df_temp = create_temperature_dependent_features(df, ['g_r_color', 'bp_rp'])
    """
    df_temp = df.copy()

    for col in color_cols:
        # Hot stars (blue colors)
        hot_name = f"hot_{col}"
        df_temp[hot_name] = (df[col] < hot_threshold).astype(int) * df[col]

        # Cool stars (red colors)
        cool_name = f"cool_{col}"
        df_temp[cool_name] = (df[col] > cool_threshold).astype(int) * df[col]

        # Mid-range
        mid_name = f"mid_{col}"
        df_temp[mid_name] = ((df[col] >= hot_threshold) & (df[col] <= cool_threshold)).astype(int) * df[col]

    return df_temp


def create_magnitude_features(
    df: pd.DataFrame,
    mag_cols: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial features from magnitudes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with magnitude columns
    mag_cols : list of str
        Magnitude column names
    degree : int, default 2
        Maximum polynomial degree

    Returns
    -------
    pd.DataFrame
        Dataframe with magnitude features

    Examples
    --------
    >>> df_mag = create_magnitude_features(df, ['gPSFMag', 'phot_g_mean_mag'])
    """
    df_mag = df.copy()

    for col in mag_cols:
        for d in range(2, degree + 1):
            feat_name = f"{col}_{d}"
            if d == 2:
                feat_name = f"{col}_squared"
            elif d == 3:
                feat_name = f"{col}_cubed"
            df_mag[feat_name] = df[col] ** d

    return df_mag


def engineer_all_features(
    df: pd.DataFrame,
    color_cols: List[str],
    mag_cols: List[str] = None,
    include_polynomials: bool = True,
    include_interactions: bool = True,
    include_log: bool = True,
    include_temp_dependent: bool = True,
    include_mag_features: bool = True
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    color_cols : list of str
        Color column names
    mag_cols : list of str, optional
        Magnitude column names
    include_polynomials : bool, default True
        Create polynomial features
    include_interactions : bool, default True
        Create interaction features
    include_log : bool, default True
        Create log features
    include_temp_dependent : bool, default True
        Create temperature-dependent features
    include_mag_features : bool, default True
        Create magnitude features

    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features

    Examples
    --------
    >>> color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
    >>> mag_cols = ['gPSFMag']
    >>> df_features = engineer_all_features(df, color_cols, mag_cols)
    """
    df_eng = df.copy()

    if include_polynomials:
        df_eng = create_polynomial_features(df_eng, color_cols, degree=3)

    if include_interactions:
        df_eng = create_interaction_features(df_eng, color_cols)

    if include_log:
        df_eng = create_log_features(df_eng, color_cols)

    if include_temp_dependent:
        df_eng = create_temperature_dependent_features(df_eng, color_cols)

    if include_mag_features and mag_cols:
        df_eng = create_magnitude_features(df_eng, mag_cols)

    return df_eng


def select_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
    score_func=f_regression
) -> Tuple[pd.DataFrame, object]:
    """
    Select k best features using univariate statistical tests.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    k : int, default 20
        Number of features to select
    score_func : callable, default f_regression
        Scoring function

    Returns
    -------
    X_selected : pd.DataFrame
        Selected features
    selector : SelectKBest
        Fitted selector object

    Examples
    --------
    >>> X_selected, selector = select_best_features(X, y, k=20)
    >>> # Get selected feature names
    >>> selected_features = X_selected.columns.tolist()
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()

    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selector


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = None
) -> pd.DataFrame:
    """
    Get feature importances from a trained model.

    Parameters
    ----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list of str
        Feature names
    top_n : int, optional
        Return only top N features

    Returns
    -------
    pd.DataFrame
        Feature importances sorted by importance

    Examples
    --------
    >>> importances = get_feature_importance(model, feature_names, top_n=10)
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    if top_n:
        importances = importances.head(top_n)

    return importances
