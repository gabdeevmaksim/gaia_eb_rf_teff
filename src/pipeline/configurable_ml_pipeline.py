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
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
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
    enc_cat = preprocess_config.get('encode_categorical', [])
    if enc_cat:
        signature_parts.append(f"encode_categorical:{','.join(sorted(enc_cat))}")
        signature_parts.append(f"categorical_encoding:{preprocess_config.get('categorical_encoding', 'onehot')}")
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
    stratify_bins = training_config.get('stratify_target_bins')
    if stratify_bins:
        signature_parts.append(f"stratify_target_bins:{stratify_bins}")
    stratify_cols = training_config.get('stratify_columns', [])
    if stratify_cols:
        signature_parts.append(f"stratify_columns:{','.join(sorted(stratify_cols))}")
    
    # Hyperparameter optimization settings (affect outcome / cache validity)
    optuna_config = model_config.get('optuna_optimization', {})
    signature_parts.append(f"optuna_method:{optuna_config.get('method', 'optuna')}")
    signature_parts.append(f"optuna_n_trials:{optuna_config.get('n_trials', 50)}")
    if optuna_config.get('max_samples') is not None:
        signature_parts.append(f"optuna_max_samples:{optuna_config['max_samples']}")
    if optuna_config.get('rf_n_jobs') is not None:
        signature_parts.append(f"optuna_rf_n_jobs:{optuna_config['rf_n_jobs']}")
    if optuna_config.get('stop_if_no_improvement_for') is not None:
        signature_parts.append(f"optuna_stop_patience:{optuna_config['stop_if_no_improvement_for']}")
    overfit_pen = optuna_config.get('overfit_penalty', 0.0)
    if overfit_pen:
        signature_parts.append(f"overfit_penalty:{overfit_pen}")
    obj_metric = optuna_config.get('objective_metric', 'mae')
    signature_parts.append(f"optuna_objective_metric:{obj_metric}")
    if obj_metric == 'crps':
        signature_parts.append(
            f"optuna_gmm_n_components:{optuna_config.get('gmm_n_components', 5)}"
        )
        signature_parts.append(
            f"optuna_crps_subsample:{optuna_config.get('crps_subsample', 5000)}"
        )
    # Successive halving options (used when method == 'halving')
    for key in [
        'factor',
        'cv',
        'n_candidates',
        'resource',
        'min_resources',
        'max_resources',
        'scoring',
        'n_jobs',
    ]:
        if optuna_config.get(key) is not None:
            signature_parts.append(f"halving_{key}:{optuna_config.get(key)}")
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
        blend_width = teff_correction.get('blend_width', 2000)
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

        # Apply correction with smooth blending to avoid a discontinuity
        self.logger.info(f"Applying polynomial correction for Teff > {threshold} K")
        self.logger.info(f"  Blend width: {blend_width} K (smooth transition {threshold}–{threshold + blend_width} K)")

        # Convert to pandas for easier manipulation
        df_pd = df.to_pandas()

        # Identify rows needing correction (inside the blend zone or above)
        needs_correction = (df_pd[target_column] > threshold) & (df_pd[target_column] != -999.0)
        n_corrected = needs_correction.sum()

        if n_corrected > 0:
            teff_original = df_pd.loc[needs_correction, target_column].values

            # Evaluate the polynomial for all affected stars
            if isinstance(coeffs_data, dict):
                poly_values = polynomial(teff_original)
            else:
                poly_values = np.zeros_like(teff_original)
                for i, coeff in enumerate(coeffs_array):
                    poly_values += coeff * (teff_original ** i)

            # Hermite smoothstep blend: ramps from 0→1 over [threshold, threshold+blend_width]
            t = np.clip((teff_original - threshold) / blend_width, 0.0, 1.0)
            blend_factor = t * t * (3.0 - 2.0 * t)

            correction_amount = poly_values - teff_original
            teff_corrected = teff_original + correction_amount * blend_factor

            # Statistics
            actual_correction = teff_corrected - teff_original
            mean_correction = actual_correction.mean()
            median_correction = np.median(actual_correction)
            n_in_blend = int(((teff_original >= threshold) & (teff_original < threshold + blend_width)).sum())

            self.logger.info(f"Correcting {n_corrected:,} stars ({n_corrected/len(df)*100:.1f}%)")
            self.logger.info(f"  Stars in blend zone ({threshold}–{threshold + blend_width} K): {n_in_blend:,}")
            self.logger.info(f"  Mean correction: {mean_correction:+.0f} K")
            self.logger.info(f"  Median correction: {median_correction:+.0f} K")
            self.logger.info(f"  Original Teff range: {teff_original.min():.0f} - {teff_original.max():.0f} K")
            self.logger.info(f"  Corrected Teff range: {teff_corrected.min():.0f} - {teff_corrected.max():.0f} K")
            self.logger.info(f"  Correction at threshold: {correction_amount[teff_original == teff_original.min()].min():+.0f} K → blended to ~0 K")

            # Create corrected column
            corrected_column = f"{target_column}_corrected"
            df_pd[corrected_column] = df_pd[target_column].astype(np.float64)
            df_pd.loc[needs_correction, corrected_column] = teff_corrected

            self.logger.info(f"Created corrected column: {corrected_column}")

            # Convert back to polars
            df = pl.from_pandas(df_pd)

            # Store correction info
            context['teff_correction_applied'] = True
            context['teff_correction_column'] = corrected_column
            context['teff_correction_stats'] = {
                'n_corrected': int(n_corrected),
                'n_in_blend_zone': n_in_blend,
                'mean_correction': float(mean_correction),
                'median_correction': float(median_correction),
                'threshold': threshold,
                'blend_width': blend_width,
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
                # Only apply numeric missing_value check/replace; skip categorical (e.g. model_type)
                cols_numeric = [c for c in cols_to_check if c in df_pd.columns and pd.api.types.is_numeric_dtype(df_pd[c])]

                # Count missing values before replacement (numeric columns only)
                missing_count = 0
                for col in cols_numeric:
                    col_missing = ((df_pd[col] == missing_value) | (np.abs(df_pd[col] - missing_value) < 1e-6)).sum()
                    missing_count += col_missing

                if missing_count > 0:
                    self.logger.info(f"Found {missing_count:,} missing value indicators ({missing_value}) in target and features")
                    for col in cols_numeric:
                        df_pd[col] = df_pd[col].replace(missing_value, np.nan)
                        close_mask = np.abs(df_pd[col].astype(float) - missing_value) < 1e-6
                        if close_mask.any():
                            df_pd.loc[close_mask, col] = np.nan
                    df = pl.from_pandas(df_pd)
                else:
                    df = pl.from_pandas(df_pd)

                # Drop nulls in target and all features (including categorical)
                df = df.drop_nulls(subset=cols_to_check)
                removed = before - len(df)
                if removed > 0:
                    self.logger.info(f"Dropped {removed:,} rows with missing values")
                elif missing_count > 0:
                    self.logger.info("Replaced missing value indicators with NaN")
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

        # Extract stratification columns (categorical) early so they survive the same filters
        training_config = model_config.get('training', {})
        strat_cols_cfg = training_config.get('stratify_columns', []) or []
        strat_col_series = {}
        for sc in strat_cols_cfg:
            if sc in df.columns:
                strat_col_series[sc] = df[sc].to_pandas()
            else:
                self.logger.warning(f"stratify_columns: '{sc}' not in dataset, ignoring")
        
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
            for sc in list(strat_col_series):
                strat_col_series[sc] = strat_col_series[sc][valid_mask].reset_index(drop=True)

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
            for sc in list(strat_col_series):
                strat_col_series[sc] = strat_col_series[sc][valid_mask].reset_index(drop=True)

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
        test_size = training_config.get('test_size', 0.2)
        random_state = training_config.get('random_state', 42)
        stratify_bins = training_config.get('stratify_target_bins', None)

        # Build stratification labels: Teff bins (optionally combined with categorical columns)
        stratify_labels = None
        min_required = max(2, int(np.ceil(1.0 / min(test_size, 1.0 - test_size))))

        if stratify_bins is not None and stratify_bins > 0:
            y_for_binning = context.get('y_original_train_full', y)
            try:
                teff_bins = pd.qcut(y_for_binning, q=stratify_bins, labels=False, duplicates='drop')
            except Exception as e:
                self.logger.warning(f"Could not create target bins: {e}; falling back to random split")
                teff_bins = None

            if teff_bins is not None:
                # Try combined stratification (bins x categorical columns) first
                if strat_col_series:
                    combined = teff_bins.astype(str)
                    for sc_name, sc_vals in strat_col_series.items():
                        combined = combined + '_' + sc_vals.astype(str)
                    n_strata = combined.nunique()
                    smallest = combined.value_counts().min()
                    if smallest >= min_required:
                        stratify_labels = combined
                        self.logger.info(
                            f"Stratified split: {teff_bins.nunique()} target bins x "
                            f"{list(strat_col_series.keys())} = {n_strata} strata, "
                            f"smallest stratum has {smallest:,} samples"
                        )
                    else:
                        self.logger.warning(
                            f"Combined stratification smallest stratum ({smallest}) < "
                            f"min required ({min_required}); falling back to target-bins-only"
                        )

                # Fall back to target-bins-only if combined didn't work or no categorical columns
                if stratify_labels is None:
                    smallest_bin = teff_bins.value_counts().min()
                    if smallest_bin >= min_required:
                        stratify_labels = teff_bins
                        self.logger.info(
                            f"Stratified split: {teff_bins.nunique()} target quantile bins "
                            f"(requested {stratify_bins}), smallest bin has {smallest_bin:,} samples"
                        )
                    else:
                        self.logger.warning(
                            f"Target-bins-only smallest bin ({smallest_bin}) < "
                            f"min required ({min_required}); falling back to random split"
                        )

        if stratify_labels is None and (stratify_bins or strat_col_series):
            self.logger.info("Using random (non-stratified) split")

        split_kwargs = dict(test_size=test_size, random_state=random_state)
        if stratify_labels is not None:
            split_kwargs['stratify'] = stratify_labels

        if sample_weight is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weight, **split_kwargs
            )
            context['sample_weights_train'] = w_train
            context['sample_weights_test'] = w_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, **split_kwargs
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

        # Store stratification labels for the training fold (reused by Optuna's inner split)
        if stratify_labels is not None:
            train_idx = X_train.index
            context['_train_stratify_labels'] = stratify_labels.iloc[train_idx].reset_index(drop=True)

        return context


class EncodeCategoricalStep(PipelineStep):
    """Encode categorical features for training (fit on train, transform train and test)."""

    def __init__(self):
        super().__init__("Encode Categorical")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        encode_cols = model_config.get('preprocessing', {}).get('encode_categorical', [])
        if not encode_cols:
            self.logger.info("No categorical columns to encode (preprocessing.encode_categorical empty)")
            return context

        X_train = context['X_train']
        X_test = context['X_test']
        feature_cols = list(context['feature_cols'])
        encoding = model_config.get('preprocessing', {}).get('categorical_encoding', 'onehot')

        # Only encode columns that exist in features
        to_encode = [c for c in encode_cols if c in feature_cols]
        if not to_encode:
            self.logger.warning(
                f"encode_categorical {encode_cols} has no overlap with feature_cols; skipping"
            )
            return context

        if encoding == 'onehot':
            encoder = OneHotEncoder(
                handle_unknown='ignore',
                drop=None,
                sparse_output=False,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )

        # Fit on training data only
        encoder.fit(X_train[to_encode])
        new_cols = encoder.get_feature_names_out(to_encode).tolist()

        # Transform train and test
        train_enc = encoder.transform(X_train[to_encode])
        test_enc = encoder.transform(X_test[to_encode])

        # Build new feature matrices: drop encoded columns, add encoded columns
        drop_cols = [c for c in feature_cols if c in to_encode]
        keep_cols = [c for c in feature_cols if c not in to_encode]
        new_feature_cols = keep_cols + new_cols

        X_train_new = X_train[keep_cols].copy()
        X_train_new[new_cols] = train_enc
        X_test_new = X_test[keep_cols].copy()
        X_test_new[new_cols] = test_enc

        # Preserve column order for downstream steps
        X_train_new = X_train_new[new_feature_cols]
        X_test_new = X_test_new[new_feature_cols]

        context['X_train'] = X_train_new
        context['X_test'] = X_test_new
        context['feature_cols'] = new_feature_cols
        context['categorical_encoder'] = encoder
        context['categorical_columns_encoded'] = to_encode

        self.logger.info(f"Encoded categorical columns: {to_encode} -> {new_cols} ({encoding})")
        self.logger.info(f"Features after encoding: {len(new_feature_cols)}")

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
        objective_metric: str = 'mae',
    ) -> None:
        """Write a text report of the Optuna optimization (all trials + best)."""
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        metric_key = 'CRPS' if objective_metric == 'crps' else 'MAE'
        if objective_metric == 'crps':
            best_line = (
                f"Best mean CRPS (validation subsample): {study.best_value:.4f}"
            )
        else:
            best_line = f"Best MAE (validation): {study.best_value:.4f}"
        lines = [
            "=" * 70,
            "Optuna Hyperparameter Optimization Report",
            "=" * 70,
            f"Model id_prefix: {id_prefix}",
            f"Signature: {signature}",
            f"Objective metric: {objective_metric}",
            f"Status: {'INTERRUPTED (user stopped)' if interrupted else 'COMPLETED'}",
            f"Trials run: {len(study.trials)}",
            f"Best trial: #{study.best_trial.number}",
            best_line,
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
                header = f"  Trial {t.number}: {metric_key}={t.value:.4f}  duration={dur_str}"
                if 'train_mae' in t.user_attrs:
                    header += (
                        f"  (train={t.user_attrs['train_mae']:.4f}"
                        f" val={t.user_attrs['val_mae']:.4f}"
                        f" gap={t.user_attrs['gap']:.4f})"
                    )
                elif 'train_crps' in t.user_attrs:
                    header += (
                        f"  (train={t.user_attrs['train_crps']:.4f}"
                        f" val={t.user_attrs['val_crps']:.4f}"
                        f" gap={t.user_attrs['gap']:.4f})"
                    )
                lines.append(header)
                for k, v in t.params.items():
                    lines.append(f"      {k}={v}")
                lines.append("")
        lines.append("=" * 70)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Optimization report written to: {report_path}")
    
    def _optimize_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              sample_weights: Optional[np.ndarray] = None,
                              max_samples: Optional[int] = None,
                              rf_n_jobs: int = -1,
                              n_trials: Optional[int] = None,
                              timeout: Optional[float] = None,
                              stop_if_no_improvement_for: Optional[int] = None,
                              report_path: Optional[Path] = None,
                              id_prefix: str = "",
                              signature: str = "",
                              search_space: Optional[Dict[str, Any]] = None,
                              stratify_labels: Optional[pd.Series] = None,
                              overfit_penalty: float = 0.0,
                              objective_metric: str = 'mae',
                              gmm_n_components: int = 5,
                              crps_subsample: int = 5000,
                              gmm_n_jobs: int = 1) -> Any:
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
            if stratify_labels is not None:
                stratify_labels = stratify_labels[valid_mask].reset_index(drop=True)
        
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
            if stratify_labels is not None:
                stratify_labels = stratify_labels[valid_mask].reset_index(drop=True)
        
        # Subsample for Optuna to reduce memory (e.g. in Docker / limited RAM)
        if max_samples is not None and len(X_train) > max_samples:
            n_original = len(X_train)
            split_kw = {'train_size': max_samples, 'random_state': 42}
            if stratify_labels is not None:
                split_kw['stratify'] = stratify_labels
            # Pack all arrays to split together, then unpack the "kept" halves
            arrays_to_split = [X_train, y_train]
            if sample_weights is not None:
                arrays_to_split.append(pd.Series(sample_weights))
            if stratify_labels is not None:
                arrays_to_split.append(stratify_labels)
            split_result = train_test_split(*arrays_to_split, **split_kw)
            # train_test_split returns [a_train, a_test, b_train, b_test, ...]
            kept = split_result[::2]   # every even index = train portion
            idx = 0
            X_train = kept[idx].reset_index(drop=True); idx += 1
            y_train = kept[idx].reset_index(drop=True); idx += 1
            if sample_weights is not None:
                sample_weights = kept[idx].values; idx += 1
            if stratify_labels is not None:
                stratify_labels = kept[idx].reset_index(drop=True); idx += 1
            self.logger.info(f"Subsampling to {len(X_train):,} samples for Optuna (from {n_original:,}) to reduce memory use")
        
        self.logger.info(f"Using {len(X_train):,} samples for Optuna optimization")
        
        # Create validation split for optimization (stratified if labels available)
        split_kwargs = {'test_size': 0.2, 'random_state': 42}
        if stratify_labels is not None:
            split_kwargs['stratify'] = stratify_labels
            self.logger.info("Optuna internal split: stratified")
        
        if sample_weights is not None:
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

        om = (objective_metric or 'mae').lower()
        if om not in ('mae', 'crps'):
            raise ValueError(
                f"Unknown objective_metric: {objective_metric!r} (expected 'mae' or 'crps')"
            )

        crps_fit_gmm = None
        crps_score = None
        val_idx = None
        train_idx = None
        n_sub_val = 0
        n_sub_tr = 0
        if om == 'crps':
            from ..mixture_density import fit_gmm_to_tree_predictions, gaussian_mixture_crps
            crps_fit_gmm = fit_gmm_to_tree_predictions
            crps_score = gaussian_mixture_crps
            n_sub_val = min(int(crps_subsample), len(y_opt_val))
            n_sub_tr = min(int(crps_subsample), len(y_opt_train))
            rng = np.random.RandomState(42)
            val_idx = rng.choice(len(y_opt_val), size=n_sub_val, replace=False)
            train_idx = rng.choice(len(y_opt_train), size=n_sub_tr, replace=False)
            self.logger.info(
                f"Optuna objective: mean CRPS (GMM components={gmm_n_components}, "
                f"val subsample n={n_sub_val}, train subsample n={n_sub_tr}; "
                f"gmm_n_jobs={gmm_n_jobs})"
            )

        if overfit_penalty > 0:
            if om == 'crps':
                self.logger.info(
                    f"Overfit penalty: {overfit_penalty} "
                    f"(objective = val_CRPS + {overfit_penalty} * |train_CRPS - val_CRPS|)"
                )
            else:
                self.logger.info(
                    f"Overfit penalty: {overfit_penalty} "
                    f"(objective = val_MAE + {overfit_penalty} * |train_MAE - val_MAE|)"
                )

        def objective(trial):
            # Suggest hyperparameters (from search_space or defaults)
            n_estimators = trial.suggest_int('n_estimators', n_est_low, n_est_high, step=n_est_step)
            max_depth = trial.suggest_int('max_depth', md_low, md_high, step=md_step)
            min_samples_split = trial.suggest_int('min_samples_split', mss_low, mss_high, step=mss_step)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', msl_low, msl_high, step=msl_step)
            max_features = trial.suggest_categorical('max_features', max_feat_choices)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=rf_n_jobs,
                verbose=0
            )
            
            if weights_opt_train is not None:
                model.fit(X_opt_train, y_opt_train, sample_weight=weights_opt_train)
            else:
                model.fit(X_opt_train, y_opt_train)

            if om == 'crps':
                Xv = X_opt_val.iloc[val_idx]
                Xv_arr = Xv.values if hasattr(Xv, 'values') else np.asarray(Xv)
                tree_preds_val = np.array([est.predict(Xv_arr) for est in model.estimators_])
                w, m, s = crps_fit_gmm(
                    tree_preds_val,
                    n_components=gmm_n_components,
                    random_state=42,
                    n_jobs=gmm_n_jobs,
                )
                yv = y_opt_val.iloc[val_idx].to_numpy()
                crps_vals = crps_score(w, m, s, yv)
                val_crps = float(np.mean(crps_vals))

                if overfit_penalty > 0:
                    Xtr = X_opt_train.iloc[train_idx]
                    Xtr_arr = Xtr.values if hasattr(Xtr, 'values') else np.asarray(Xtr)
                    tree_preds_tr = np.array(
                        [est.predict(Xtr_arr) for est in model.estimators_]
                    )
                    w2, m2, s2 = crps_fit_gmm(
                        tree_preds_tr,
                        n_components=gmm_n_components,
                        random_state=42,
                        n_jobs=gmm_n_jobs,
                    )
                    ytr = y_opt_train.iloc[train_idx].to_numpy()
                    crps_tr_vals = crps_score(w2, m2, s2, ytr)
                    train_crps = float(np.mean(crps_tr_vals))
                    gap = abs(train_crps - val_crps)
                    trial.set_user_attr('train_crps', train_crps)
                    trial.set_user_attr('val_crps', val_crps)
                    trial.set_user_attr('gap', float(gap))
                    return val_crps + overfit_penalty * gap

                return val_crps

            # Evaluate on validation set (MAE)
            val_mae = mean_absolute_error(y_opt_val, model.predict(X_opt_val))

            if overfit_penalty > 0:
                train_mae = mean_absolute_error(y_opt_train, model.predict(X_opt_train))
                gap = abs(train_mae - val_mae)
                trial.set_user_attr('train_mae', float(train_mae))
                trial.set_user_attr('val_mae', float(val_mae))
                trial.set_user_attr('gap', float(gap))
                return val_mae + overfit_penalty * gap

            return val_mae
        
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
                study,
                report_path,
                id_prefix,
                signature,
                interrupted=interrupted,
                objective_metric=om,
            )

        if interrupted:
            if complete_trials:
                _label = 'CRPS' if om == 'crps' else 'MAE'
                self.logger.info(
                    f"Best so far: {_label}={study.best_value:.4f} "
                    f"(trial {study.best_trial.number})"
                )
            return (best_params, True)
        if not complete_trials:
            return best_params
        if om == 'crps':
            self.logger.info(
                f"✓ Optimization complete. Best mean CRPS (val subsample): "
                f"{study.best_value:.2f}"
            )
        else:
            self.logger.info(f"✓ Optimization complete. Best MAE: {study.best_value:.2f}")
        for param, value in best_params.items():
            if param not in ['random_state', 'n_jobs', 'verbose']:
                self.logger.info(f"  {param}: {value}")
        return best_params

    def _optimize_with_halving(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        sample_weights: Optional[np.ndarray] = None,
        rf_n_jobs: int = 1,
        factor: int = 3,
        cv: int = 3,
        n_candidates: Any = "exhaust",
        resource: str = "n_samples",
        # sklearn constraint: n_candidates and min_resources cannot both be 'exhaust'
        min_resources: Any = "smallest",
        max_resources: Any = "auto",
        scoring: str = "neg_mean_absolute_error",
        n_jobs: int = -1,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Successive-halving hyperparameter search using HalvingRandomSearchCV.

        This approach scales the number of training rows up across iterations
        (resource='n_samples'), which is usually much more memory-stable than
        training full-size forests for every candidate.
        """
        space = search_space or {}

        def _as_list(key: str, default: List[Any]) -> List[Any]:
            val = space.get(key)
            if val is None:
                return default
            if isinstance(val, list) and len(val) in (2, 3) and isinstance(val[0], int) and isinstance(val[1], int):
                low, high = int(val[0]), int(val[1])
                step = int(val[2]) if len(val) == 3 else (50 if key == "n_estimators" else 1)
                step = max(step, 1)
                return list(range(low, high + 1, step))
            if isinstance(val, list):
                return val
            return default

        param_dist = {
            "n_estimators": _as_list("n_estimators", list(range(100, 1001, 50))),
            "max_depth": _as_list("max_depth", list(range(5, 51, 1))),
            "min_samples_split": _as_list("min_samples_split", list(range(2, 21, 1))),
            "min_samples_leaf": _as_list("min_samples_leaf", list(range(1, 11, 1))),
            "max_features": _as_list("max_features", ["sqrt", "log2", None]),
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=rf_n_jobs, verbose=0)

        search = HalvingRandomSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_candidates=n_candidates,
            factor=factor,
            resource=resource,
            min_resources=min_resources,
            max_resources=max_resources,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=3,
        )

        if sample_weights is not None:
            search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            search.fit(X_train, y_train)

        best_params = dict(search.best_params_)
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1  # full training still uses full parallelism
        best_params["verbose"] = 0

        if best_params.get("max_features") is None:
            best_params["max_features"] = "sqrt"

        self.logger.info(f"Halving search best score ({scoring}): {search.best_score_}")
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
        
        # No cache found -- warn and show how to skip
        hp = model_config.get('hyperparameters', {})
        self.logger.warning(
            f"No cached hyperparameters for signature {signature} in {cache_file.name}."
        )
        if hp:
            self.logger.warning(
                "If you already have optimized hyperparameters, you can stop "
                "the pipeline (Ctrl+C), set 'optuna_optimization.enabled: false' "
                "in the config file, and re-run to use the hyperparameters block directly."
            )
        method = optuna_config.get('method', 'optuna')
        if method not in ('optuna', 'halving'):
            raise ValueError(
                f"Unknown optuna_optimization.method: {method} (expected 'optuna' or 'halving')"
            )
        objective_metric = (optuna_config.get('objective_metric') or 'mae').lower()
        if objective_metric not in ('mae', 'crps'):
            raise ValueError(
                "optuna_optimization.objective_metric must be 'mae' or 'crps' "
                f"(got {objective_metric!r})"
            )
        if method == 'halving' and objective_metric == 'crps':
            raise ValueError(
                "objective_metric 'crps' is only supported with optuna_optimization.method: "
                "'optuna'. Use method: optuna, or set objective_metric: mae for halving."
            )
        self.logger.info(f"Proceeding with hyperparameter optimization (method={method})...")
        self.logger.info(f"Optuna objective metric: {objective_metric}")
        
        # Get optimization settings from config
        n_trials = optuna_config.get('n_trials', self.n_trials)
        timeout = optuna_config.get('timeout', self.timeout)
        max_samples = optuna_config.get('max_samples')  # None = use full data; int = subsample for Optuna only
        # Default rf_n_jobs: optuna uses -1 by default; halving uses 1 by default (lower memory)
        rf_n_jobs = optuna_config.get('rf_n_jobs', 1 if method == 'halving' else -1)
        stop_if_no_improvement_for = optuna_config.get('stop_if_no_improvement_for')  # int or None
        
        # Report path: reports/optuna_reports/optuna_<id_prefix>_<timestamp>.txt
        id_prefix = model_config.get('model', {}).get('id_prefix', 'model')
        config = context.get('config') or get_config()
        reports_dir = Path(config.get_path('reports'))
        optuna_reports_dir = reports_dir / 'optuna_reports'
        optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = optuna_reports_dir / f"optuna_{id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        search_space = optuna_config.get('search_space')

        # Reuse stratification labels computed in PrepareTrainTestFromConfigStep
        optuna_stratify_labels = context.get('_train_stratify_labels')

        if method == 'halving':
            # Successive halving handles resource ramp-up itself; ignore Optuna-specific settings.
            result = self._optimize_with_halving(
                X_train,
                y_train,
                sample_weights=sample_weights,
                rf_n_jobs=rf_n_jobs,
                factor=int(optuna_config.get('factor', 3)),
                cv=int(optuna_config.get('cv', 3)),
                n_candidates=optuna_config.get('n_candidates', 'exhaust'),
                resource=optuna_config.get('resource', 'n_samples'),
                min_resources=optuna_config.get('min_resources', 'smallest'),
                max_resources=optuna_config.get('max_resources', 'auto'),
                scoring=optuna_config.get('scoring', 'neg_mean_absolute_error'),
                n_jobs=int(optuna_config.get('n_jobs', -1)),
                search_space=search_space,
            )
        else:
            result = self._optimize_with_optuna(
                X_train,
                y_train,
                sample_weights,
                max_samples=max_samples,
                rf_n_jobs=rf_n_jobs,
                n_trials=n_trials,
                timeout=timeout,
                stop_if_no_improvement_for=stop_if_no_improvement_for,
                report_path=report_path,
                id_prefix=id_prefix,
                signature=signature,
                search_space=search_space,
                stratify_labels=optuna_stratify_labels,
                overfit_penalty=float(optuna_config.get('overfit_penalty', 0.0)),
                objective_metric=objective_metric,
                gmm_n_components=int(optuna_config.get('gmm_n_components', 5)),
                crps_subsample=int(optuna_config.get('crps_subsample', 5000)),
                gmm_n_jobs=int(optuna_config.get('gmm_n_jobs', 1)),
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

        # Per-tree predictions for test set (used for uncertainty + GMM)
        tree_preds_transformed = None
        y_pred_test_uncertainty = None
        if hasattr(model, 'estimators_') and model.estimators_:
            feature_cols = context['feature_cols']
            X_test_np = (
                X_test[feature_cols].to_numpy()
                if isinstance(X_test, pd.DataFrame)
                else np.asarray(X_test)
            )
            tree_preds_transformed = np.array(
                [est.predict(X_test_np) for est in model.estimators_]
            )  # shape (n_trees, n_samples)

        # Inverse transform predictions if target was transformed
        target_transform = context.get('target_transform', 'none')
        if target_transform == 'log':
            self.logger.info("Inverse transforming predictions (10^y)")
            y_pred_train = 10 ** y_pred_train
            y_pred_test = 10 ** y_pred_test
            y_train = 10 ** y_train
            y_test = 10 ** y_test
            if tree_preds_transformed is not None:
                unc_transformed = np.std(tree_preds_transformed, axis=0)
                y_pred_test_uncertainty = y_pred_test * unc_transformed * np.log(10)
                tree_preds_transformed = 10 ** tree_preds_transformed
        elif target_transform == 'log2':
            self.logger.info("Inverse transforming predictions (2^y)")
            y_pred_train = 2 ** y_pred_train
            y_pred_test = 2 ** y_pred_test
            y_train = 2 ** y_train
            y_test = 2 ** y_test
            if tree_preds_transformed is not None:
                unc_transformed = np.std(tree_preds_transformed, axis=0)
                y_pred_test_uncertainty = y_pred_test * unc_transformed * np.log(2)
                tree_preds_transformed = 2 ** tree_preds_transformed
        elif target_transform == 'ln':
            self.logger.info("Inverse transforming predictions (e^y)")
            y_pred_train = np.exp(y_pred_train)
            y_pred_test = np.exp(y_pred_test)
            y_train = np.exp(y_train)
            y_test = np.exp(y_test)
            if tree_preds_transformed is not None:
                unc_transformed = np.std(tree_preds_transformed, axis=0)
                y_pred_test_uncertainty = y_pred_test * unc_transformed
                tree_preds_transformed = np.exp(tree_preds_transformed)
        else:
            if tree_preds_transformed is not None:
                y_pred_test_uncertainty = np.std(tree_preds_transformed, axis=0)

        if y_pred_test_uncertainty is not None:
            context['y_pred_test_uncertainty'] = y_pred_test_uncertainty
            self.logger.info(
                f"  Test set inner uncertainty: median={np.median(y_pred_test_uncertainty):.1f} K, "
                f"mean={np.mean(y_pred_test_uncertainty):.1f} K"
            )

        # Fit Gaussian mixture to tree predictions (in original scale)
        if tree_preds_transformed is not None:
            from ..mixture_density import (
                fit_gmm_to_tree_predictions,
                fit_gmm_to_tree_predictions_bic,
                gaussian_mixture_crps,
                gaussian_mixture_pit,
            )
            val_config = model_config.get('validation', {})
            gmm_select_k = val_config.get('gmm_select_k')  # None or "bic"
            rs = context.get('model_config', {}).get('training', {}).get('random_state', 42)

            if gmm_select_k == 'bic':
                k_max = int(val_config.get('gmm_k_max', 8))
                self.logger.info(
                    f"Fitting GMM (BIC selection, K=1..{k_max}) to tree predictions "
                    f"({tree_preds_transformed.shape[0]} trees, "
                    f"{tree_preds_transformed.shape[1]} samples)..."
                )
                gmm_weights, gmm_means, gmm_sigmas, best_ks = (
                    fit_gmm_to_tree_predictions_bic(
                        tree_preds_transformed,
                        k_max=k_max,
                        random_state=rs,
                    )
                )
                context['gmm_best_ks'] = best_ks
                self.logger.info(
                    f"  BIC-selected K: mean={np.mean(best_ks):.1f}, "
                    f"median={np.median(best_ks):.0f}, "
                    f"min={best_ks.min()}, max={best_ks.max()}"
                )
            else:
                n_components = int(val_config.get('gmm_n_components', 5))
                self.logger.info(
                    f"Fitting {n_components}-component GMM to tree predictions "
                    f"({tree_preds_transformed.shape[0]} trees, "
                    f"{tree_preds_transformed.shape[1]} samples)..."
                )
                gmm_weights, gmm_means, gmm_sigmas = fit_gmm_to_tree_predictions(
                    tree_preds_transformed,
                    n_components=n_components,
                    random_state=rs,
                )

            context['gmm_weights'] = gmm_weights
            context['gmm_means'] = gmm_means
            context['gmm_sigmas'] = gmm_sigmas

            # CRPS and PIT
            crps_values = gaussian_mixture_crps(gmm_weights, gmm_means, gmm_sigmas, y_test)
            pit_values = gaussian_mixture_pit(gmm_weights, gmm_means, gmm_sigmas, y_test)
            context['crps_values'] = crps_values
            context['pit_values'] = pit_values

            mean_crps = float(np.mean(crps_values))
            median_crps = float(np.median(crps_values))
            context['crps_stats'] = {'mean': mean_crps, 'median': median_crps}

            self.logger.info(f"  GMM CRPS:  mean={mean_crps:.1f}, median={median_crps:.1f}")
            self.logger.info(f"  GMM PIT:   mean={np.mean(pit_values):.3f} (ideal=0.5), "
                             f"std={np.std(pit_values):.3f} (ideal~0.289)")

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

        # Add probabilistic metrics (GMM-based CRPS/PIT)
        if context.get('crps_stats'):
            metadata['crps_stats'] = context['crps_stats']
            pit = context.get('pit_values')
            if pit is not None:
                metadata['pit_stats'] = {
                    'mean': float(np.mean(pit)),
                    'std': float(np.std(pit)),
                }
            gmm_meta = {}
            best_ks = context.get('gmm_best_ks')
            if best_ks is not None:
                gmm_meta['select_k'] = 'bic'
                gmm_meta['k_max'] = int(context['gmm_weights'].shape[1])
                gmm_meta['k_mean'] = float(np.mean(best_ks))
                gmm_meta['k_median'] = int(np.median(best_ks))
                gmm_meta['k_min'] = int(best_ks.min())
                gmm_meta['k_max_used'] = int(best_ks.max())
            else:
                gmm_meta['select_k'] = 'fixed'
                gmm_meta['n_components'] = int(context['gmm_weights'].shape[1])
            metadata['gmm_config'] = gmm_meta

        # Add Teff correction info if applied
        if context.get('teff_correction_applied', False):
            metadata['teff_correction'] = context.get('teff_correction_stats', {})
            metadata['teff_correction']['target_column'] = model_config.get('teff_correction', {}).get('target_column', 'teff_gaia')
            metadata['teff_correction']['corrected_column'] = context.get('teff_correction_column', 'unknown')

        # Save categorical encoder if used
        if context.get('categorical_encoder') is not None:
            encoder_file = models_dir / f"{model_id}_categorical_encoder.pkl"
            joblib.dump(context['categorical_encoder'], encoder_file)
            metadata['categorical_encoder_path'] = encoder_file.name
            metadata['categorical_columns_encoded'] = context.get('categorical_columns_encoded', [])
            metadata['categorical_encoding'] = model_config.get('preprocessing', {}).get('categorical_encoding', 'onehot')
            self.logger.info(f"✓ Saved categorical encoder: {encoder_file.name}")

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
            f.write(f"Features: {len(feature_cols)}\n")

            # List all feature columns for clarity (including any EBV features)
            if feature_cols:
                f.write("Feature columns:\n")
                for feat in feature_cols:
                    f.write(f"  - {feat}\n")

                # Explicitly highlight extinction features if present
                ebv_features = [f for f in feature_cols if 'ebv' in f.lower()]
                if ebv_features:
                    f.write("\nExtinction feature columns (E(B-V)):\n")
                    for feat in ebv_features:
                        f.write(f"  - {feat}\n")

            f.write("\n")

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

            if context.get('crps_stats'):
                crps_stats = context['crps_stats']
                best_ks = context.get('gmm_best_ks')
                if best_ks is not None:
                    f.write(
                        f"\nPROBABILISTIC METRICS (BIC-selected GMM, "
                        f"K=1..{context['gmm_weights'].shape[1]}, "
                        f"median K={int(np.median(best_ks))}):\n"
                    )
                else:
                    n_comp = context['gmm_weights'].shape[1]
                    f.write(f"\nPROBABILISTIC METRICS ({n_comp}-component GMM):\n")
                f.write(f"  CRPS mean:   {crps_stats['mean']:.1f} K\n")
                f.write(f"  CRPS median: {crps_stats['median']:.1f} K\n")
                pit = context.get('pit_values')
                if pit is not None:
                    f.write(f"  PIT mean:    {np.mean(pit):.3f} (ideal=0.500)\n")
                    f.write(f"  PIT std:     {np.std(pit):.3f} (ideal~0.289)\n")

        self.logger.info(f"✓ Saved summary: {summary_file.name}")

        # Save predictions with original features (for color-temp plots)
        # Note: y_test and y_pred_test are already inverse-transformed if needed
        predictions_df = pd.DataFrame({
            'y_true': context['y_test'],
            'y_pred': context['y_pred_test']
        })
        if context.get('y_pred_test_uncertainty') is not None:
            predictions_df['y_pred_uncertainty'] = context['y_pred_test_uncertainty']
            self.logger.info("  (includes inner uncertainty column: y_pred_uncertainty)")

        # GMM mixture parameters
        if context.get('gmm_weights') is not None:
            n_comp = context['gmm_weights'].shape[1]
            for k in range(n_comp):
                predictions_df[f'gmm_weight_{k}'] = context['gmm_weights'][:, k]
                predictions_df[f'gmm_mean_{k}'] = context['gmm_means'][:, k]
                predictions_df[f'gmm_sigma_{k}'] = context['gmm_sigmas'][:, k]
            if context.get('gmm_best_ks') is not None:
                predictions_df['gmm_n_components'] = context['gmm_best_ks']
            predictions_df['crps'] = context['crps_values']
            predictions_df['pit'] = context['pit_values']
            bic_note = " (BIC-selected K)" if context.get('gmm_best_ks') is not None else ""
            self.logger.info(f"  (includes {n_comp}-component GMM{bic_note}, CRPS, PIT)")

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
            EncodeCategoricalStep(),
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
