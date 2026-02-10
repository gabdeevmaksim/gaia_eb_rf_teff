"""
Configurable machine learning training pipeline for temperature prediction.

This pipeline uses YAML configuration files to define model variants,
eliminating the need for separate training scripts for each model.

Usage:
    # Train using model config
    pipeline = ConfigurableMLPipeline('config/models/gaia_2mass_ir.yaml')
    context = pipeline.run()

    # Or use CLI
    python pipeline.py --ml-config config/models/gaia_2mass_ir.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
import joblib
import json
import yaml
import logging
import hashlib

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep
from ..features import engineer_all_features
from ..config import get_config


def _generate_model_signature(model_config: Dict[str, Any]) -> str:
    """
    Generate a unique signature/hash for a model configuration.
    
    This signature is used to determine if hyperparameters have already
    been optimized for a model with the same settings.
    
    Parameters
    ----------
    model_config : dict
        Model configuration dictionary
        
    Returns
    -------
    str
        SHA256 hash of the model signature
    """
    # Create signature from key model settings
    signature_parts = []
    
    # Data source
    data_config = model_config.get('data', {})
    signature_parts.append(f"source_file:{data_config.get('source_file', '')}")
    signature_parts.append(f"target:{data_config.get('target', '')}")
    
    # Features (sorted for consistency)
    features = sorted(data_config.get('features', []))
    signature_parts.append(f"features:{','.join(features)}")
    
    # Preprocessing
    preprocess_config = model_config.get('preprocessing', {})
    signature_parts.append(f"missing_value:{preprocess_config.get('missing_value', -999.0)}")
    filters = preprocess_config.get('filters', {})
    if filters:
        # Sort filters for consistency
        filter_str = ','.join([f"{k}:{v}" for k, v in sorted(filters.items())])
        signature_parts.append(f"filters:{filter_str}")
    
    # Feature engineering
    feature_eng_config = model_config.get('feature_engineering', {})
    signature_parts.append(f"enabled:{feature_eng_config.get('enabled', False)}")
    if feature_eng_config.get('enabled', False):
        signature_parts.append(f"include_polynomials:{feature_eng_config.get('include_polynomials', True)}")
        signature_parts.append(f"include_interactions:{feature_eng_config.get('include_interactions', True)}")
        signature_parts.append(f"include_log:{feature_eng_config.get('include_log', True)}")
        signature_parts.append(f"include_temp_dependent:{feature_eng_config.get('include_temp_dependent', True)}")
    
    # Clustering
    clustering_config = model_config.get('clustering', {})
    signature_parts.append(f"clustering_enabled:{clustering_config.get('enabled', False)}")
    if clustering_config.get('enabled', False):
        signature_parts.append(f"clustering_method:{clustering_config.get('method', 'gmm')}")
        signature_parts.append(f"n_clusters:{clustering_config.get('n_clusters', 5)}")
    
    # Target transform
    signature_parts.append(f"target_transform:{model_config.get('target_transform', 'none')}")
    
    # Training settings that affect optimization
    training_config = model_config.get('training', {})
    signature_parts.append(f"test_size:{training_config.get('test_size', 0.2)}")
    signature_parts.append(f"random_state:{training_config.get('random_state', 42)}")
    
    # Optuna settings (affect optimization outcome / cache validity)
    optuna_config = model_config.get('optuna_optimization', {})
    signature_parts.append(f"optuna_n_trials:{optuna_config.get('n_trials', 50)}")
    if optuna_config.get('max_samples') is not None:
        signature_parts.append(f"optuna_max_samples:{optuna_config['max_samples']}")
    if optuna_config.get('stop_if_no_improvement_for') is not None:
        signature_parts.append(f"optuna_stop_patience:{optuna_config['stop_if_no_improvement_for']}")
    search_space = optuna_config.get('search_space')
    if search_space:
        signature_parts.append(f"search_space:{json.dumps(search_space, sort_keys=True)}")
    
    # Create hash
    signature_string = '|'.join(signature_parts)
    signature_hash = hashlib.sha256(signature_string.encode()).hexdigest()[:16]
    
    return signature_hash


class LoadModelConfigStep(PipelineStep):
    """Load model configuration from YAML file."""

    def __init__(self, config_path: str):
        super().__init__("Load Model Configuration")
        self.config_path = config_path

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")

        with open(config_file, 'r') as f:
            model_config = yaml.safe_load(f)

        self.logger.info(f"Loaded model config: {config_file.name}")
        self.logger.info(f"Model: {model_config['model']['name']}")
        self.logger.info(f"Description: {model_config['model']['description']}")
        
        # Check if there are cached hyperparameters for this model signature
        # (This will be checked again in Optuna step, but we can log it here)
        signature = _generate_model_signature(model_config)
        hyperparam_cache = model_config.get('hyperparameter_cache', {})
        if signature in hyperparam_cache:
            cached_info = hyperparam_cache[signature]
            self.logger.info(f"Found cached hyperparameters (optimized at: {cached_info.get('optimized_at', 'unknown')})")

        context['model_config'] = model_config
        return context


class LoadDataFromConfigStep(PipelineStep):
    """Load training data based on model configuration."""

    def __init__(self):
        super().__init__("Load Training Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        config = context['config']

        # Get data source
        source_file = model_config['data']['source_file']

        # Check if source_location is specified (raw/processed), default to processed
        source_location = model_config['data'].get('source_location', 'processed')
        data_dir = Path(config.get_path(source_location))
        data_path = data_dir / source_file

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        # Load data based on format
        if data_path.suffix == '.parquet':
            df = pl.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pl.read_csv(data_path)
        elif data_path.suffix == '.fits':
            # Load FITS file using astropy, convert to polars
            from astropy.table import Table
            table = Table.read(str(data_path))
            df = pl.from_pandas(table.to_pandas())
            self.logger.info(f"Loaded FITS file from {source_location}/")
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        self.logger.info(f"Loaded {len(df):,} samples from {source_file}")

        context['raw_data'] = df
        return context


class ApplyTeffCorrectionStep(PipelineStep):
    """Apply polynomial Teff correction if configured."""

    def __init__(self):
        super().__init__("Apply Teff Correction")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['raw_data']
        model_config = context['model_config']
        config = context['config']

        # Check if Teff correction is enabled
        teff_correction = model_config.get('teff_correction', {})

        if not teff_correction.get('enabled', False):
            self.logger.info("Teff correction disabled - skipping")
            context['teff_correction_applied'] = False
            return context

        self.logger.info("Applying Teff correction to target variable")

        # Get configuration
        target_column = teff_correction.get('target_column', 'teff_gaia')
        threshold = teff_correction.get('threshold', 10000)
        coeffs_file = teff_correction.get('coefficients_file', 'teff_correction_coeffs_deg2.pkl')

        # Check if target column exists
        if target_column not in df.columns:
            self.logger.warning(f"Target column '{target_column}' not found - skipping correction")
            context['teff_correction_applied'] = False
            return context

        # Load correction coefficients
        coeffs_path = Path(config.get_path('data_root')) / coeffs_file
        if not coeffs_path.exists():
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Teff correction coefficients file not found!\n"
                f"{'='*70}\n"
                f"Expected location: {coeffs_path}\n"
                f"File name: {coeffs_file}\n\n"
                f"To download the correction coefficients, run:\n"
                f"  python scripts/download_datasets.py --datasets correction\n\n"
                f"Or if using Docker:\n"
                f"  docker compose run --rm data --datasets correction\n\n"
                f"The file will be downloaded to: {Path(config.get_path('data_root'))}\n"
                f"{'='*70}\n"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(
                f"Teff correction coefficients not found: {coeffs_path}\n"
                f"Download with: python scripts/download_datasets.py --datasets correction"
            )

        self.logger.info(f"Loading correction coefficients from: {coeffs_path}")
        coeffs_data = joblib.load(coeffs_path)

        # Handle both dict format (with Polynomial object) and array format
        if isinstance(coeffs_data, dict):
            polynomial = coeffs_data['polynomial']
            degree = coeffs_data.get('degree', 2)
            self.logger.info(f"Using numpy Polynomial object (degree {degree})")
        else:
            # Legacy format: simple array of coefficients
            coeffs_array = coeffs_data
            degree = len(coeffs_array) - 1
            self.logger.info(f"Using coefficient array (degree {degree})")

        # Apply correction
        self.logger.info(f"Applying polynomial correction for Teff > {threshold} K")

        # Convert to pandas for easier manipulation
        df_pd = df.to_pandas()

        # Identify rows needing correction
        needs_correction = (df_pd[target_column] > threshold) & (df_pd[target_column] != -999.0)
        n_corrected = needs_correction.sum()

        if n_corrected > 0:
            teff_original = df_pd.loc[needs_correction, target_column].values

            # Apply correction based on format
            if isinstance(coeffs_data, dict):
                # Use Polynomial object's __call__ method
                teff_corrected = polynomial(teff_original)
            else:
                # Apply polynomial correction manually: Teff_corrected = sum(coeffs[i] * Teff^i)
                teff_corrected = np.zeros_like(teff_original)
                for i, coeff in enumerate(coeffs_array):
                    teff_corrected += coeff * (teff_original ** i)

            # Calculate correction statistics
            mean_correction = (teff_corrected - teff_original).mean()
            median_correction = np.median(teff_corrected - teff_original)

            self.logger.info(f"Correcting {n_corrected:,} stars ({n_corrected/len(df)*100:.1f}%)")
            self.logger.info(f"  Mean correction: {mean_correction:+.0f} K")
            self.logger.info(f"  Median correction: {median_correction:+.0f} K")
            self.logger.info(f"  Original Teff range: {teff_original.min():.0f} - {teff_original.max():.0f} K")
            self.logger.info(f"  Corrected Teff range: {teff_corrected.min():.0f} - {teff_corrected.max():.0f} K")

            # Create corrected column
            corrected_column = f"{target_column}_corrected"
            df_pd[corrected_column] = df_pd[target_column].copy()
            df_pd.loc[needs_correction, corrected_column] = teff_corrected
            
            # Ensure corrected values are float64 to avoid dtype warnings
            df_pd[corrected_column] = df_pd[corrected_column].astype(np.float64)

            self.logger.info(f"Created corrected column: {corrected_column}")

            # Convert back to polars
            df = pl.from_pandas(df_pd)

            # Store correction info
            context['teff_correction_applied'] = True
            context['teff_correction_column'] = corrected_column
            context['teff_correction_stats'] = {
                'n_corrected': int(n_corrected),
                'mean_correction': float(mean_correction),
                'median_correction': float(median_correction),
                'threshold': threshold,
                'degree': degree
            }
        else:
            self.logger.info("No stars above threshold - no correction applied")
            context['teff_correction_applied'] = False

        context['raw_data'] = df
        return context


class PreprocessDataStep(PipelineStep):
    """Preprocess data based on configuration."""

    def __init__(self):
        super().__init__("Preprocess Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['raw_data']
        model_config = context['model_config']
        config = context['config']

        preprocessing = model_config.get('preprocessing', {})
        target = model_config['data']['target']
        features = model_config['data'].get('features', [])

        # Step 1: Apply value filters from config (e.g., temperature range, color limits)
        filters = preprocessing.get('filters', {})
        if filters:
            self.logger.info(f"Applying {len(filters)} value filters from config")
            initial_count = len(df)

            for col, (min_val, max_val) in filters.items():
                if col in df.columns:
                    before = len(df)
                    df = df.filter((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
                    removed = before - len(df)
                    self.logger.info(f"  {col}: [{min_val}, {max_val}] → removed {removed:,} rows")
                else:
                    self.logger.warning(
                        f"  Filter column '{col}' not in dataset (columns: {df.columns[:15]}...); "
                        f"filter [{min_val}, {max_val}] skipped."
                    )

            self.logger.info(f"After value filters: {len(df):,} samples ({len(df)/initial_count*100:.1f}% retained)")

        # Step 2: Handle missing value indicators (-999.0) and drop rows with missing/NaN values
        if preprocessing.get('drop_missing', True):
            before = len(df)
            missing_value = preprocessing.get('missing_value', -999.0)
            
            # Replace missing value indicators with NaN for target and features
            cols_to_check = [target] + features
            cols_to_check = [c for c in cols_to_check if c in df.columns]
            
            if not cols_to_check:
                self.logger.warning(f"Target '{target}' or features not found in dataframe. Available columns: {df.columns[:10]}")
            else:
                # Convert to pandas for more robust -999 replacement, then back to polars
                df_pd = df.to_pandas()
                
                # Count missing values before replacement
                missing_count = 0
                for col in cols_to_check:
                    if col in df_pd.columns:
                        # Check for exact match and also for values very close to missing_value (handles float precision issues)
                        col_missing = ((df_pd[col] == missing_value) | (np.abs(df_pd[col] - missing_value) < 1e-6)).sum()
                        missing_count += col_missing
                
                if missing_count > 0:
                    self.logger.info(f"Found {missing_count:,} missing value indicators ({missing_value}) in target and features")
                    for col in cols_to_check:
                        if col in df_pd.columns:
                            # Replace missing value indicator with NaN (handles both exact match and close values)
                            df_pd[col] = df_pd[col].replace(missing_value, np.nan)
                            # Also replace values very close to missing_value
                            close_mask = np.abs(df_pd[col] - missing_value) < 1e-6
                            if close_mask.any():
                                df_pd.loc[close_mask, col] = np.nan
                    
                    df = pl.from_pandas(df_pd)
                    
                    # Drop nulls in target and features
                    df = df.drop_nulls(subset=cols_to_check)
                    removed = before - len(df)
                    self.logger.info(f"Replaced {missing_value} with NaN and dropped {removed:,} rows with missing values")
                else:
                    # Still check for existing NaN values
                    df = pl.from_pandas(df_pd)
                    df = df.drop_nulls(subset=cols_to_check)
                    removed = before - len(df)
                    if removed > 0:
                        self.logger.info(f"Dropped {removed:,} rows with NaN values")
                    else:
                        self.logger.info("No missing values found")

        self.logger.info(f"After preprocessing: {len(df):,} samples")

        context['clean_data'] = df
        return context


class EngineerFeaturesFromConfigStep(PipelineStep):
    """Engineer features if enabled in configuration."""

    def __init__(self):
        super().__init__("Engineer Features")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['clean_data']
        model_config = context['model_config']

        feature_config = model_config.get('feature_engineering', {})

        if feature_config.get('enabled', False):
            self.logger.info("Feature engineering enabled")

            # Convert to pandas for feature engineering
            df_pd = df.to_pandas()

            color_cols = feature_config.get('color_cols', [])
            mag_cols = feature_config.get('mag_cols', [])

            self.logger.info(f"Color columns: {color_cols}")
            self.logger.info(f"Magnitude columns: {mag_cols}")

            # Engineer features
            df_features = engineer_all_features(
                df_pd,
                color_cols=color_cols,
                mag_cols=mag_cols
            )

            self.logger.info(f"Engineered features: {df_features.shape[1]} columns "
                           f"(from {df.shape[1]})")

            # Convert back to polars
            df_result = pl.from_pandas(df_features)
        else:
            self.logger.info("Feature engineering disabled - using raw features")
            df_result = df

            # Apply custom features from config if specified
            custom_features = feature_config.get('custom_features', {})
            if custom_features:
                self.logger.info(f"Adding {len(custom_features)} custom features from config")
                df_pd = df_result.to_pandas()

                for feature_name, feature_expr in custom_features.items():
                    try:
                        df_pd[feature_name] = df_pd.eval(feature_expr)
                        self.logger.info(f"  Added: {feature_name} = {feature_expr}")
                    except Exception as e:
                        self.logger.warning(f"  Failed to add {feature_name}: {e}")

                df_result = pl.from_pandas(df_pd)
                self.logger.info(f"Features after custom additions: {df_result.shape[1]} columns")

        context['feature_data'] = df_result
        return context


class PrepareTrainTestFromConfigStep(PipelineStep):
    """Prepare train/test split based on configuration."""

    def __init__(self):
        super().__init__("Prepare Train/Test Split")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['feature_data']
        model_config = context['model_config']

        # Get target and features
        target = model_config['data']['target']
        exclude_cols = model_config['data'].get('exclude_columns', [])
        specified_features = model_config['data'].get('features')

        # Determine feature columns
        if specified_features:
            # Use specified features
            feature_cols = [c for c in specified_features if c in df.columns]
            self.logger.info(f"Using {len(feature_cols)} specified features")
        else:
            # Auto-detect: all columns except target and excluded
            all_exclude = [target] + exclude_cols
            feature_cols = [c for c in df.columns if c not in all_exclude]
            self.logger.info(f"Auto-detected {len(feature_cols)} features")

        # Extract features and target
        X = df.select(feature_cols).to_pandas()
        y = df[target].to_pandas()
        
        # Get missing value indicator for filtering
        preprocessing = model_config.get('preprocessing', {})
        missing_value = preprocessing.get('missing_value', -999.0)
        
        # Filter out missing values and invalid values BEFORE any transformation
        # This handles both -999 (missing) and <=0 (invalid for log)
        # Use more robust comparison that handles float precision issues
        is_missing = (y == missing_value) | (np.abs(y - missing_value) < 1e-6)
        valid_mask = ~is_missing & (y > 0) & ~y.isna() & ~np.isinf(y)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            self.logger.warning(f"Filtering {n_invalid:,} invalid values (missing={missing_value}, <=0, NaN, or Inf) before transformation")
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)

        # Apply target transformation if specified
        target_transform = model_config.get('target_transform', 'none')
        if target_transform == 'log':
            self.logger.info("Applying log transformation to target variable")
            y_original = y.copy()
            y = np.log10(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original  # Store for reference
            context['target_transform'] = 'log'
        elif target_transform == 'log2':
            self.logger.info("Applying log2 transformation to target variable")
            y_original = y.copy()
            y = np.log2(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original
            context['target_transform'] = 'log2'
        elif target_transform == 'ln':
            self.logger.info("Applying natural log transformation to target variable")
            y_original = y.copy()
            y = np.log(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original
            context['target_transform'] = 'ln'
        else:
            context['target_transform'] = 'none'
        
        # Final safety check: remove any NaN/Inf values that might have been created
        valid_mask = ~(y.isna() | np.isinf(y))
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            self.logger.warning(f"Removing {n_invalid:,} samples with NaN/Inf in transformed target")
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)
            if 'y_original_train_full' in context:
                context['y_original_train_full'] = context['y_original_train_full'][valid_mask].reset_index(drop=True)

        # Check for sample weights
        preprocessing = model_config.get('preprocessing', {})
        sample_weight = None

        if preprocessing.get('use_sample_weights', False):
            weight_col = preprocessing.get('weight_column', 'sample_weight')
            if weight_col in df.columns:
                sample_weight = df[weight_col].to_pandas().values
                self.logger.info(f"Using sample weights from column: {weight_col}")
                self.logger.info(f"  Weight range: [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")
            else:
                self.logger.warning(f"Sample weight column '{weight_col}' not found, ignoring")

        # Train/test split
        training_config = model_config.get('training', {})
        test_size = training_config.get('test_size', 0.2)
        random_state = training_config.get('random_state', 42)

        if sample_weight is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weight,
                test_size=test_size,
                random_state=random_state
            )
            context['sample_weights_train'] = w_train
            context['sample_weights_test'] = w_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

        self.logger.info(f"Train: {len(X_train):,} samples")
        self.logger.info(f"Test:  {len(X_test):,} samples")
        self.logger.info(f"Features: {len(feature_cols)}")

        # Log top features for inspection
        if len(feature_cols) <= 10:
            self.logger.info(f"Feature list: {feature_cols}")
        else:
            self.logger.info(f"First 5 features: {feature_cols[:5]}")

        context['X_train'] = X_train
        context['X_test'] = X_test
        context['y_train'] = y_train
        context['y_test'] = y_test
        context['feature_cols'] = feature_cols

        return context


class AddClusteringFeaturesStep(PipelineStep):
    """Add cluster membership probabilities as features.

    Supports multiple clustering methods:
    - gmm: Gaussian Mixture Model
    - bayesian_gmm: Bayesian GMM (auto-selects number of clusters)
    - kmeans: KMeans with soft distance-based probabilities
    """

    def __init__(self):
        super().__init__("Add Clustering Features")

    def _fit_gmm(self, X_train_scaled, clustering_config, random_state):
        """Fit Gaussian Mixture Model."""
        n_clusters = clustering_config.get('n_clusters', 5)

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=clustering_config.get('covariance_type', 'full'),
            n_init=clustering_config.get('n_init', 10),
            max_iter=clustering_config.get('max_iter', 200),
            reg_covar=clustering_config.get('reg_covar', 1e-6),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"GMM converged: {model.converged_}")
        self.logger.info(f"GMM iterations: {model.n_iter_}")

        return model, n_clusters

    def _fit_bayesian_gmm(self, X_train_scaled, clustering_config, random_state):
        """Fit Bayesian Gaussian Mixture Model (auto-selects clusters)."""
        n_clusters = clustering_config.get('n_clusters', 10)  # Max clusters

        model = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type=clustering_config.get('covariance_type', 'full'),
            n_init=clustering_config.get('n_init', 5),
            max_iter=clustering_config.get('max_iter', 200),
            reg_covar=clustering_config.get('reg_covar', 1e-6),
            weight_concentration_prior_type=clustering_config.get('weight_prior', 'dirichlet_process'),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"Bayesian GMM converged: {model.converged_}")
        self.logger.info(f"Bayesian GMM iterations: {model.n_iter_}")

        # Count effective clusters (weight > 0.01)
        effective_clusters = (model.weights_ > 0.01).sum()
        self.logger.info(f"Effective clusters (weight > 1%): {effective_clusters} / {n_clusters}")

        for i, w in enumerate(model.weights_):
            if w > 0.01:
                self.logger.info(f"  Cluster {i}: weight = {w:.4f}")

        return model, n_clusters

    def _fit_kmeans(self, X_train_scaled, clustering_config, random_state):
        """Fit KMeans with soft distance-based probabilities."""
        n_clusters = clustering_config.get('n_clusters', 5)

        model = KMeans(
            n_clusters=n_clusters,
            n_init=clustering_config.get('n_init', 10),
            max_iter=clustering_config.get('max_iter', 300),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"KMeans inertia: {model.inertia_:.2f}")
        self.logger.info(f"KMeans iterations: {model.n_iter_}")

        return model, n_clusters

    def _get_kmeans_probabilities(self, model, X_scaled):
        """Convert KMeans distances to soft probabilities."""
        # Get distances to all centroids
        distances = model.transform(X_scaled)

        # Convert distances to similarity (inverse distance)
        # Add small epsilon to avoid division by zero
        similarities = 1.0 / (distances + 1e-8)

        # Normalize to probabilities
        probs = similarities / similarities.sum(axis=1, keepdims=True)

        return probs

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        clustering_config = model_config.get('clustering', {})

        if not clustering_config.get('enabled', False):
            self.logger.info("Clustering features disabled - skipping")
            return context

        method = clustering_config.get('method', 'gmm')
        random_state = model_config.get('training', {}).get('random_state', 42)

        self.logger.info(f"Adding clustering features using method: {method}")

        X_train = context['X_train']
        X_test = context['X_test']
        feature_cols = context['feature_cols']

        # Standardize features for clustering
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        X_test_scaled = scaler.transform(X_test.astype(np.float64))

        # Fit clustering model based on method
        if method == 'gmm':
            model, n_clusters = self._fit_gmm(X_train_scaled, clustering_config, random_state)
            train_probs = model.predict_proba(X_train_scaled)
            test_probs = model.predict_proba(X_test_scaled)

        elif method == 'bayesian_gmm':
            model, n_clusters = self._fit_bayesian_gmm(X_train_scaled, clustering_config, random_state)
            train_probs = model.predict_proba(X_train_scaled)
            test_probs = model.predict_proba(X_test_scaled)

        elif method == 'kmeans':
            model, n_clusters = self._fit_kmeans(X_train_scaled, clustering_config, random_state)
            train_probs = self._get_kmeans_probabilities(model, X_train_scaled)
            test_probs = self._get_kmeans_probabilities(model, X_test_scaled)

        else:
            raise ValueError(f"Unknown clustering method: {method}. "
                           f"Supported: gmm, bayesian_gmm, kmeans")

        # Add probability features to dataframes
        prob_col_names = [f'cluster_prob_{i}' for i in range(n_clusters)]

        X_train_enh = np.hstack([X_train.values, train_probs])
        X_test_enh = np.hstack([X_test.values, test_probs])

        feature_cols_enh = list(feature_cols) + prob_col_names

        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_enh, columns=feature_cols_enh)
        X_test_df = pd.DataFrame(X_test_enh, columns=feature_cols_enh)

        self.logger.info(f"Added {n_clusters} cluster probability features")
        self.logger.info(f"Total features: {len(feature_cols_enh)} (was {len(feature_cols)})")

        # Analyze clusters by target value
        train_labels = train_probs.argmax(axis=1)
        y_train = context['y_train']
        if hasattr(y_train, 'values'):
            y_train_arr = y_train.values
        else:
            y_train_arr = y_train

        self.logger.info("Cluster statistics:")
        for i in range(n_clusters):
            mask = train_labels == i
            n_obj = mask.sum()
            if n_obj > 0:
                target_mean = y_train_arr[mask].mean()
                target_std = y_train_arr[mask].std()
                self.logger.info(f"  Cluster {i}: {n_obj:,} objects ({n_obj/len(train_labels)*100:.1f}%) "
                               f"- Target mean: {target_mean:.1f} +/- {target_std:.1f}")

        # Store models for saving
        context['clustering_model'] = model
        context['clustering_scaler'] = scaler
        context['clustering_method'] = method
        context['n_clusters'] = n_clusters

        # Update context
        context['X_train'] = X_train_df
        context['X_test'] = X_test_df
        context['feature_cols'] = feature_cols_enh

        return context


class OptunaOptimizeHyperparametersStep(PipelineStep):
    """
    Optimize hyperparameters using Optuna.
    
    This step:
    1. Generates a model signature based on data source, features, and settings
    2. Checks if optimized hyperparameters already exist in config for this signature
    3. If not found, runs Optuna optimization
    4. Saves optimized hyperparameters to the config file
    5. Updates the model_config in context with optimized hyperparameters
    """
    
    def __init__(self, n_trials: int = 50, timeout: Optional[float] = None):
        """
        Parameters
        ----------
        n_trials : int, default=50
            Number of Optuna trials to run
        timeout : float, optional
            Maximum time in seconds for optimization (None = no limit)
        """
        super().__init__("Optimize Hyperparameters")
        self.n_trials = n_trials
        self.timeout = timeout
        
    def _get_hyperparameter_cache_path(self, model_config_path: str) -> Path:
        """
        Path to the shared hyperparameter cache file (never overwrites model config).
        Placed in config/hyperparameter_cache.yaml.
        """
        path = Path(model_config_path).resolve()
        # path is .../config/models/foo.yaml -> parent is config/models, parent.parent is config
        config_dir = path.parent.parent
        return config_dir / "hyperparameter_cache.yaml"

    def _check_existing_hyperparameters(
        self,
        model_config: Dict[str, Any],
        signature: str,
        cache_file: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if hyperparameters already exist for this model signature.
        Looks in the dedicated cache file first, then in model_config (backward compat).
        """
        # 1. Dedicated cache file (preferred)
        if cache_file is not None and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = yaml.safe_load(f)
                if cache and signature in cache:
                    self.logger.info(f"Found cached hyperparameters for signature {signature} in {cache_file.name}")
                    return cache[signature].get('hyperparameters')
            except Exception as e:
                self.logger.warning(f"Could not read hyperparameter cache file: {e}")
        # 2. In-model cache (backward compatibility)
        hyperparam_cache = model_config.get('hyperparameter_cache', {})
        if signature in hyperparam_cache:
            self.logger.info(f"Found cached hyperparameters for signature {signature} (from config)")
            return hyperparam_cache[signature].get('hyperparameters')
        return None

    def _save_hyperparameters_to_cache_file(
        self,
        cache_file: Path,
        signature: str,
        hyperparams: Dict[str, Any],
        n_trials_used: int,
    ) -> None:
        """
        Save optimized hyperparameters to the dedicated cache file only.
        Does not modify the original model config file.
        """
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.warning(f"Could not read existing cache file, creating new: {e}")
        cache[signature] = {
            'hyperparameters': hyperparams,
            'optimized_at': datetime.now().isoformat(),
            'n_trials': n_trials_used,
        }
        with open(cache_file, 'w') as f:
            yaml.dump(cache, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f"Saved optimized hyperparameters to {cache_file}")

    def _write_optuna_report(
        self,
        study,
        report_path: Path,
        id_prefix: str,
        signature: str,
        interrupted: bool,
    ) -> None:
        """Write a text report of the Optuna optimization (all trials + best)."""
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "=" * 70,
            "Optuna Hyperparameter Optimization Report",
            "=" * 70,
            f"Model id_prefix: {id_prefix}",
            f"Signature: {signature}",
            f"Status: {'INTERRUPTED (user stopped)' if interrupted else 'COMPLETED'}",
            f"Trials run: {len(study.trials)}",
            f"Best trial: #{study.best_trial.number}",
            f"Best MAE (validation): {study.best_value:.4f}",
            "",
            "Best hyperparameters:",
            "-" * 40,
        ]
        for k, v in study.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.extend([
            "",
            "All trials (trial_number, value, duration_sec, params):",
            "-" * 40,
        ])
        for t in study.trials:
            if t.state.name == "COMPLETE":
                dur = t.duration.total_seconds() if t.duration else None
                dur_str = f"{dur:.1f}s" if dur is not None else "?"
                lines.append(f"  Trial {t.number}: MAE={t.value:.4f}  duration={dur_str}")
                for k, v in t.params.items():
                    lines.append(f"      {k}={v}")
                lines.append("")
        lines.append("=" * 70)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Optimization report written to: {report_path}")
    
    def _optimize_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              sample_weights: Optional[np.ndarray] = None,
                              max_samples: Optional[int] = None,
                              n_trials: Optional[int] = None,
                              timeout: Optional[float] = None,
                              stop_if_no_improvement_for: Optional[int] = None,
                              report_path: Optional[Path] = None,
                              id_prefix: str = "",
                              signature: str = "",
                              search_space: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run Optuna optimization to find best hyperparameters.
        On KeyboardInterrupt, writes report and returns (best_params, True).
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )
        
        # Filter out NaN values in target and features
        valid_mask = ~(y_train.isna() | np.isinf(y_train))
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            self.logger.warning(f"Removing {n_invalid:,} samples with NaN/Inf in target before optimization")
            X_train = X_train[valid_mask].reset_index(drop=True)
            y_train = y_train[valid_mask].reset_index(drop=True)
            if sample_weights is not None:
                sample_weights = sample_weights[valid_mask]
        
        # Also check for NaN in features
        feature_nan_mask = X_train.isna().any(axis=1)
        if feature_nan_mask.any():
            n_feature_nan = feature_nan_mask.sum()
            self.logger.warning(f"Removing {n_feature_nan:,} samples with NaN in features before optimization")
            valid_mask = ~feature_nan_mask
            X_train = X_train[valid_mask].reset_index(drop=True)
            y_train = y_train[valid_mask].reset_index(drop=True)
            if sample_weights is not None:
                sample_weights = sample_weights[valid_mask]
        
        # Subsample for Optuna to reduce memory (e.g. in Docker / limited RAM)
        if max_samples is not None and len(X_train) > max_samples:
            n_original = len(X_train)
            split_kw = {'train_size': max_samples, 'random_state': 42}
            if sample_weights is not None:
                X_train, _, y_train, _, sample_weights, _ = train_test_split(
                    X_train, y_train, sample_weights, **split_kw
                )
            else:
                X_train, _, y_train, _ = train_test_split(X_train, y_train, **split_kw)
            self.logger.info(f"Subsampling to {len(X_train):,} samples for Optuna (from {n_original:,}) to reduce memory use")
        
        self.logger.info(f"Using {len(X_train):,} samples for Optuna optimization")
        
        # Create validation split for optimization
        split_kwargs = {'test_size': 0.2, 'random_state': 42}
        
        if sample_weights is not None:
            # train_test_split can handle multiple arrays
            X_opt_train, X_opt_val, y_opt_train, y_opt_val, weights_opt_train, weights_opt_val = train_test_split(
                X_train, y_train, sample_weights, **split_kwargs
            )
        else:
            X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
                X_train, y_train, **split_kwargs
            )
            weights_opt_train = None
        
        # Resolve search space: config overrides defaults
        space = search_space or {}
        def _int_range(key: str, default_low: int, default_high: int, default_step: int = 1):
            val = space.get(key)
            if val is None:
                return default_low, default_high, default_step
            if isinstance(val, list) and len(val) >= 2:
                low, high = int(val[0]), int(val[1])
                step = int(val[2]) if len(val) >= 3 else (50 if key == 'n_estimators' else 1)
                return low, high, step
            return default_low, default_high, default_step
        def _cat_choices(key: str, default: List[Any]):
            val = space.get(key)
            if val is None or not isinstance(val, list):
                return default
            return val
        n_est_low, n_est_high, n_est_step = _int_range('n_estimators', 100, 1000, 50)
        md_low, md_high, md_step = _int_range('max_depth', 5, 50, 1)
        mss_low, mss_high, mss_step = _int_range('min_samples_split', 2, 20, 1)
        msl_low, msl_high, msl_step = _int_range('min_samples_leaf', 1, 10, 1)
        max_feat_choices = _cat_choices('max_features', ['sqrt', 'log2', None])

        def objective(trial):
            # Suggest hyperparameters (from search_space or defaults)
            n_estimators = trial.suggest_int('n_estimators', n_est_low, n_est_high, step=n_est_step)
            max_depth = trial.suggest_int('max_depth', md_low, md_high, step=md_step)
            min_samples_split = trial.suggest_int('min_samples_split', mss_low, mss_high, step=mss_step)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', msl_low, msl_high, step=msl_step)
            max_features = trial.suggest_categorical('max_features', max_feat_choices)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            if weights_opt_train is not None:
                model.fit(X_opt_train, y_opt_train, sample_weight=weights_opt_train)
            else:
                model.fit(X_opt_train, y_opt_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_opt_val)
            mae = mean_absolute_error(y_opt_val, y_pred)
            
            return mae
        
        # Early stopping: stop after N trials without improving the best value
        callbacks = []
        if stop_if_no_improvement_for is not None:
            patience = stop_if_no_improvement_for
            def _early_stop_callback(study, trial):
                if trial.number - study.best_trial.number >= patience:
                    self.logger.info(f"Stopping: no improvement for {patience} trials (best at trial {study.best_trial.number})")
                    study.stop()
            callbacks.append(_early_stop_callback)
            self.logger.info(f"Early stopping: will stop if no improvement for {patience} trials")

        # Create study
        study = optuna.create_study(direction='minimize')
        trials = n_trials if n_trials is not None else self.n_trials
        time_limit = timeout if timeout is not None else self.timeout
        self.logger.info(f"Starting Optuna optimization (max {trials} trials)...")

        interrupted = False
        try:
            study.optimize(
                objective,
                n_trials=trials,
                timeout=time_limit,
                callbacks=callbacks if callbacks else None,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            interrupted = True
            self.logger.warning("Optimization interrupted by user (Ctrl+C). Saving best result so far...")

        # Build best_params (may be from incomplete run)
        complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not complete_trials:
            if interrupted:
                self.logger.error("No completed trials; nothing to save. Exiting.")
                raise KeyboardInterrupt
            best_params = {}
        else:
            best_params = study.best_params.copy()
            # Convert max_features None to string for YAML compatibility
            if best_params.get('max_features') is None:
                best_params['max_features'] = 'sqrt'
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_params['verbose'] = 0

        # Write report (always, so you have a print of the optimisation)
        if report_path is not None and complete_trials:
            self._write_optuna_report(
                study, report_path, id_prefix, signature, interrupted=interrupted
            )

        if interrupted:
            if complete_trials:
                self.logger.info(f"Best so far: MAE={study.best_value:.4f} (trial {study.best_trial.number})")
            return (best_params, True)
        if not complete_trials:
            return best_params
        self.logger.info(f"✓ Optimization complete. Best MAE: {study.best_value:.2f}")
        for param, value in best_params.items():
            if param not in ['random_state', 'n_jobs', 'verbose']:
                self.logger.info(f"  {param}: {value}")
        return best_params
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        config_file = context['config_file']
        X_train = context['X_train']
        y_train = context['y_train']
        sample_weights = context.get('sample_weights_train')
        
        # Check if optimization is enabled
        optuna_config = model_config.get('optuna_optimization', {})
        if not optuna_config.get('enabled', False):
            self.logger.info("Optuna optimization disabled - using hyperparameters from config")
            return context
        
        # Generate model signature
        signature = _generate_model_signature(model_config)
        self.logger.info(f"Model signature: {signature}")
        
        # Dedicated cache file (never overwrite model config)
        cache_file = self._get_hyperparameter_cache_path(config_file)
        existing_hyperparams = self._check_existing_hyperparameters(
            model_config, signature, cache_file=cache_file
        )
        
        if existing_hyperparams is not None:
            self.logger.info("Using existing optimized hyperparameters from cache")
            model_config['hyperparameters'] = existing_hyperparams
            context['model_config'] = model_config
            return context
        
        # Run optimization
        self.logger.info("No cached hyperparameters found - running Optuna optimization...")
        
        # Get optimization settings from config
        n_trials = optuna_config.get('n_trials', self.n_trials)
        timeout = optuna_config.get('timeout', self.timeout)
        max_samples = optuna_config.get('max_samples')  # None = use full data; int = subsample for Optuna only
        stop_if_no_improvement_for = optuna_config.get('stop_if_no_improvement_for')  # int or None
        
        # Report path: reports/optuna_reports/optuna_<id_prefix>_<timestamp>.txt
        id_prefix = model_config.get('model', {}).get('id_prefix', 'model')
        config = context.get('config') or get_config()
        reports_dir = Path(config.get_path('reports'))
        optuna_reports_dir = reports_dir / 'optuna_reports'
        optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = optuna_reports_dir / f"optuna_{id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        search_space = optuna_config.get('search_space')
        result = self._optimize_with_optuna(
            X_train, y_train, sample_weights,
            max_samples=max_samples,
            n_trials=n_trials,
            timeout=timeout,
            stop_if_no_improvement_for=stop_if_no_improvement_for,
            report_path=report_path,
            id_prefix=id_prefix,
            signature=signature,
            search_space=search_space
        )
        
        # Handle interrupt: save best to cache, log, then exit
        if isinstance(result, tuple):
            optimized_hyperparams, was_interrupted = result
            if was_interrupted and optimized_hyperparams:
                n_trials_used = len(context.get('_optuna_trials_run', 0))  # we don't have it; use n_trials as proxy
                self._save_hyperparameters_to_cache_file(
                    cache_file, signature, optimized_hyperparams,
                    n_trials if n_trials is not None else self.n_trials
                )
                self.logger.info(f"Best hyperparameters saved to cache. Report: {report_path}")
            raise KeyboardInterrupt
        
        optimized_hyperparams = result
        
        # Save to dedicated cache file only (never overwrite model config)
        n_trials_used = n_trials if n_trials is not None else self.n_trials
        self._save_hyperparameters_to_cache_file(
            cache_file, signature, optimized_hyperparams, n_trials_used
        )
        
        # Update model_config in context
        model_config['hyperparameters'] = optimized_hyperparams
        context['model_config'] = model_config
        
        return context


class TrainModelFromConfigStep(PipelineStep):
    """Train Random Forest model using configuration."""

    def __init__(self):
        super().__init__("Train Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context['X_train']
        y_train = context['y_train']
        model_config = context['model_config']

        # Get hyperparameters
        hyperparams = model_config.get('hyperparameters', {})

        # Handle max_features (can be string or number)
        max_features = hyperparams.get('max_features', 'log2')
        if isinstance(max_features, str) and max_features.isdigit():
            max_features = int(max_features)

        model_params = {
            'n_estimators': hyperparams.get('n_estimators', 300),
            'max_depth': hyperparams.get('max_depth', 20),
            'min_samples_split': hyperparams.get('min_samples_split', 5),
            'min_samples_leaf': hyperparams.get('min_samples_leaf', 4),
            'max_features': max_features,
            'random_state': hyperparams.get('random_state', 42),
            'n_jobs': hyperparams.get('n_jobs', -1),
            'verbose': 0
        }

        self.logger.info("Training Random Forest with parameters:")
        for param, value in model_params.items():
            if param != 'verbose':
                self.logger.info(f"  {param}: {value}")

        # Create and train model
        model = RandomForestRegressor(**model_params)

        # Use sample weights if available
        sample_weights = context.get('sample_weights_train')
        if sample_weights is not None:
            self.logger.info("Training with sample weights")
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        self.logger.info("✓ Model trained successfully")

        context['model'] = model
        context['hyperparameters'] = model_params
        return context


class EvaluateModelFromConfigStep(PipelineStep):
    """Evaluate model performance."""

    def __init__(self):
        super().__init__("Evaluate Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        X_train = context['X_train']
        X_test = context['X_test']
        y_train = context['y_train']
        y_test = context['y_test']

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Inverse transform predictions if target was transformed
        target_transform = context.get('target_transform', 'none')
        if target_transform == 'log':
            self.logger.info("Inverse transforming predictions (10^y)")
            y_pred_train = 10 ** y_pred_train
            y_pred_test = 10 ** y_pred_test
            y_train = 10 ** y_train
            y_test = 10 ** y_test
        elif target_transform == 'log2':
            self.logger.info("Inverse transforming predictions (2^y)")
            y_pred_train = 2 ** y_pred_train
            y_pred_test = 2 ** y_pred_test
            y_train = 2 ** y_train
            y_test = 2 ** y_test
        elif target_transform == 'ln':
            self.logger.info("Inverse transforming predictions (e^y)")
            y_pred_train = np.exp(y_pred_train)
            y_pred_test = np.exp(y_pred_test)
            y_train = np.exp(y_train)
            y_test = np.exp(y_test)

        # Calculate metrics (on original scale)
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)

        # Update context with inverse-transformed values for saving
        context['y_train'] = y_train
        context['y_test'] = y_test

        self.logger.info("Performance Metrics:")
        self.logger.info("  TRAIN SET:")
        self.logger.info(f"    MAE:  {train_metrics['mae']:.0f} K")
        self.logger.info(f"    RMSE: {train_metrics['rmse']:.0f} K")
        self.logger.info(f"    R²:   {train_metrics['r2']:.4f}")
        self.logger.info(f"    Within 10%: {train_metrics['within_10_percent']:.1f}%")

        self.logger.info("  TEST SET:")
        self.logger.info(f"    MAE:  {test_metrics['mae']:.0f} K")
        self.logger.info(f"    RMSE: {test_metrics['rmse']:.0f} K")
        self.logger.info(f"    R²:   {test_metrics['r2']:.4f}")
        self.logger.info(f"    Within 10%: {test_metrics['within_10_percent']:.1f}%")

        # Feature importance
        feature_importance = dict(zip(
            context['feature_cols'],
            model.feature_importances_
        ))

        # Sort by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("Top 10 Features by Importance:")
        for feature, importance in top_features:
            self.logger.info(f"    {feature}: {importance:.4f}")

        context['y_pred_train'] = y_pred_train
        context['y_pred_test'] = y_pred_test
        context['train_metrics'] = train_metrics
        context['test_metrics'] = test_metrics
        context['feature_importance'] = feature_importance

        return context

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Relative error
        relative_errors = np.abs((y_true - y_pred) / y_true)
        mean_rel_error = np.mean(relative_errors) * 100

        # Percentage within thresholds
        within_5 = (relative_errors <= 0.05).sum() / len(y_true) * 100
        within_10 = (relative_errors <= 0.10).sum() / len(y_true) * 100
        within_20 = (relative_errors <= 0.20).sum() / len(y_true) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_relative_error': float(mean_rel_error),
            'within_5_percent': float(within_5),
            'within_10_percent': float(within_10),
            'within_20_percent': float(within_20),
            'n_samples': len(y_true)
        }


class SaveModelFromConfigStep(PipelineStep):
    """Save trained model and artifacts."""

    def __init__(self):
        super().__init__("Save Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        model_config = context['model_config']
        test_metrics = context['test_metrics']
        train_metrics = context['train_metrics']
        feature_cols = context['feature_cols']
        feature_importance = context['feature_importance']
        config = context['config']

        # Create model ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_prefix = model_config['model']['id_prefix']
        model_id = f"{id_prefix}_{timestamp}"

        models_dir = Path(config.get_path('models', ensure_exists=True))

        # Save model
        model_file = models_dir / f"{model_id}.pkl"
        joblib.dump(model, model_file)
        self.logger.info(f"✓ Saved model: {model_file.name}")

        # Save metadata
        metadata = {
            'model_id': model_id,
            'timestamp': timestamp,
            'model_name': model_config['model']['name'],
            'description': model_config['model']['description'],
            'model_type': 'RandomForestRegressor',
            'n_features': len(feature_cols),
            'features': feature_cols,
            'data_source': model_config['data']['source_file'],
            'target': model_config['data']['target'],
            'target_transform': context.get('target_transform', 'none'),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'hyperparameters': context['hyperparameters'],
            'config_file': str(context.get('config_file', 'unknown')),
            'data_config': model_config['data']  # Save full data config for validation
        }

        # Add Teff correction info if applied
        if context.get('teff_correction_applied', False):
            metadata['teff_correction'] = context.get('teff_correction_stats', {})
            metadata['teff_correction']['target_column'] = model_config.get('teff_correction', {}).get('target_column', 'teff_gaia')
            metadata['teff_correction']['corrected_column'] = context.get('teff_correction_column', 'unknown')

        metadata_file = models_dir / f"{model_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✓ Saved metadata: {metadata_file.name}")

        # Save summary text file
        summary_file = models_dir / f"{model_id}_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model: {model_config['model']['name']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model ID: {model_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Description: {model_config['model']['description']}\n\n")

            f.write(f"Data Source: {model_config['data']['source_file']}\n")
            f.write(f"Target: {model_config['data']['target']}\n")
            target_transform = context.get('target_transform', 'none')
            if target_transform != 'none':
                f.write(f"Target Transform: {target_transform}\n")
            f.write(f"Features: {len(feature_cols)}\n\n")

            f.write("TRAIN SET PERFORMANCE:\n")
            f.write(f"  MAE:  {train_metrics['mae']:.1f} K\n")
            f.write(f"  RMSE: {train_metrics['rmse']:.1f} K\n")
            f.write(f"  R²:   {train_metrics['r2']:.4f}\n")
            f.write(f"  Within 10%: {train_metrics['within_10_percent']:.1f}%\n\n")

            f.write("TEST SET PERFORMANCE:\n")
            f.write(f"  MAE:  {test_metrics['mae']:.1f} K\n")
            f.write(f"  RMSE: {test_metrics['rmse']:.1f} K\n")
            f.write(f"  R²:   {test_metrics['r2']:.4f}\n")
            f.write(f"  Within 10%: {test_metrics['within_10_percent']:.1f}%\n\n")

            f.write("TOP 10 FEATURES:\n")
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                f.write(f"  {i:2d}. {feature:30s} {importance:.4f}\n")

        self.logger.info(f"✓ Saved summary: {summary_file.name}")

        # Save predictions with original features (for color-temp plots)
        # Note: y_test and y_pred_test are already inverse-transformed if needed
        predictions_df = pd.DataFrame({
            'y_true': context['y_test'],
            'y_pred': context['y_pred_test']
        })

        # Add original color features to predictions for validation plots
        # Get the original features from X_test (before any feature engineering)
        X_test_df = context['X_test']
        if isinstance(X_test_df, pd.DataFrame):
            # Try to add raw feature columns (colors) that might be useful for plotting
            data_features = model_config['data'].get('features', [])
            for feat in data_features:
                if feat in X_test_df.columns:
                    predictions_df[feat] = X_test_df[feat].values

        pred_file = models_dir / f"{model_id}_test_predictions.parquet"
        predictions_df.to_parquet(pred_file)
        self.logger.info(f"✓ Saved predictions: {pred_file.name}")

        # Save clustering models if they were used
        if 'clustering_model' in context:
            method = context.get('clustering_method', 'gmm')
            cluster_file = models_dir / f"{model_id}_clustering_{method}.pkl"
            scaler_file = models_dir / f"{model_id}_clustering_scaler.pkl"

            joblib.dump(context['clustering_model'], cluster_file)
            joblib.dump(context['clustering_scaler'], scaler_file)

            self.logger.info(f"✓ Saved clustering model ({method}): {cluster_file.name}")
            self.logger.info(f"✓ Saved clustering scaler: {scaler_file.name}")

            # Update metadata with clustering info
            metadata['clustering'] = {
                'method': method,
                'n_clusters': context['n_clusters'],
                'model_file': str(cluster_file.name),
                'scaler_file': str(scaler_file.name)
            }

            # Re-save metadata with clustering info
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        context['model_id'] = model_id
        context['model_file'] = str(model_file)
        context['metadata_file'] = str(metadata_file)
        context['summary_file'] = str(summary_file)

        return context


class ConfigurableMLPipeline(Pipeline):
    """
    Configurable ML training pipeline that uses YAML configuration files.

    This eliminates the need for separate training scripts for each model variant.
    All model variations can be trained using different config files.

    Usage
    -----
    >>> # Train using config file
    >>> pipeline = ConfigurableMLPipeline('config/models/gaia_2mass_ir.yaml')
    >>> context = pipeline.run()
    >>> print(f"Model saved: {context['model_id']}")

    >>> # Or from CLI
    >>> python pipeline.py --ml-config config/models/gaia_2mass_ir.yaml

    Parameters
    ----------
    model_config_path : str
        Path to model configuration YAML file
    """

    def __init__(self, model_config_path: str):
        # Store config path for later use
        self.model_config_path = model_config_path

        steps = [
            LoadModelConfigStep(model_config_path),
            LoadDataFromConfigStep(),
            ApplyTeffCorrectionStep(),  # Apply Teff correction before preprocessing
            PreprocessDataStep(),
            EngineerFeaturesFromConfigStep(),
            PrepareTrainTestFromConfigStep(),
            AddClusteringFeaturesStep(),
            OptunaOptimizeHyperparametersStep(),  # Optimize hyperparameters before training
            TrainModelFromConfigStep(),
            EvaluateModelFromConfigStep(),
            SaveModelFromConfigStep(),
        ]

        # Extract model name from config for pipeline name
        config_file = Path(model_config_path)
        pipeline_name = f"ML Training Pipeline ({config_file.stem})"

        super().__init__(pipeline_name, steps)

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run the pipeline with initial context."""
        context = {
            'config': get_config(),
            'config_file': self.model_config_path
        }
        return super().run(context)
