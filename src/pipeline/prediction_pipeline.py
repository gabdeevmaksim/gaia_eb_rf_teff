"""
Prediction pipeline for temperature estimation.

This pipeline loads a trained model and applies it to new observations,
ensuring the same feature engineering is applied as during training.

Usage:
    # Using config file
    pipeline = PredictionPipeline('config/prediction/predict_all_sources.yaml')
    context = pipeline.run()

    # Or from CLI
    python pipeline.py predict --config config/prediction/predict_all_sources.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import joblib
import yaml
import json

import polars as pl
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep
from ..features import engineer_all_features
from ..config import get_config


class LoadPredictionConfigStep(PipelineStep):
    """Load prediction configuration from YAML file."""

    def __init__(self, config_path: str):
        super().__init__("Load Prediction Configuration")
        self.config_path = config_path

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Prediction config not found: {config_file}")

        with open(config_file, 'r') as f:
            pred_config = yaml.safe_load(f)

        self.logger.info(f"Loaded prediction config: {config_file.name}")
        self.logger.info(f"Model: {pred_config['model']['model_file']}")
        self.logger.info(f"Data source: {pred_config['data']['source_file']}")

        context['prediction_config'] = pred_config
        return context


class LoadTrainedModelStep(PipelineStep):
    """Load trained model and its metadata."""

    def __init__(self):
        super().__init__("Load Trained Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pred_config = context['prediction_config']
        config = context['config']

        # Get model file (supports wildcards like rf_model_*.pkl)
        model_pattern = pred_config['model']['model_file']
        models_dir = Path(config.get_path('models'))

        # Find model file
        if '*' in model_pattern:
            # Wildcard pattern - find most recent
            model_files = sorted(models_dir.glob(model_pattern))
            if not model_files:
                raise FileNotFoundError(f"No model found matching: {model_pattern}")
            model_file = model_files[-1]  # Most recent
            self.logger.info(f"Found {len(model_files)} matching models, using most recent")
        else:
            # Exact filename
            model_file = models_dir / model_pattern
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")

        self.logger.info(f"Loading model: {model_file.name}")

        # Load model
        model = joblib.load(model_file)

        # Load metadata if exists
        metadata_file = model_file.with_name(f"{model_file.stem}_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"Loaded metadata: {metadata.get('model_name', 'Unknown')}")
            self.logger.info(f"  Features: {metadata.get('n_features', 'Unknown')}")
            self.logger.info(f"  Test MAE: {metadata.get('test_metrics', {}).get('mae', 'Unknown'):.0f} K")
        else:
            self.logger.warning(f"Metadata file not found: {metadata_file.name}")
            metadata = {}

        context['model'] = model
        context['model_metadata'] = metadata
        context['model_file'] = str(model_file)
        context['model_id'] = model_file.stem

        return context


class LoadPredictionDataStep(PipelineStep):
    """Load data for prediction."""

    def __init__(self):
        super().__init__("Load Prediction Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pred_config = context['prediction_config']
        config = context['config']

        # Get data source
        source_file = pred_config['data']['source_file']
        source_location = pred_config['data'].get('source_location', 'processed')
        data_dir = Path(config.get_path(source_location))
        data_path = data_dir / source_file

        if not data_path.exists():
            raise FileNotFoundError(f"Prediction data not found: {data_path}")

        # Load data
        if data_path.suffix == '.parquet':
            df = pl.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pl.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        self.logger.info(f"Loaded {len(df):,} objects from {source_file}")

        # Apply filters if specified
        filters = pred_config['data'].get('filters', {})
        if filters:
            self.logger.info("Applying data filters...")
            filter_mask = pl.lit(True)

            for filter_name, filter_value in filters.items():
                # Handle special filter syntax
                if filter_name.endswith('_is_null'):
                    # Column null check: col_name_is_null: true/false
                    col_name = filter_name.replace('_is_null', '')
                    if filter_value:
                        filter_mask = filter_mask & pl.col(col_name).is_null()
                        self.logger.info(f"  Filter: {col_name} IS NULL")
                    else:
                        filter_mask = filter_mask & pl.col(col_name).is_not_null()
                        self.logger.info(f"  Filter: {col_name} IS NOT NULL")

                elif filter_name.endswith('_eq'):
                    # Equality check: col_name_eq: value
                    col_name = filter_name.replace('_eq', '')
                    filter_mask = filter_mask & (pl.col(col_name) == filter_value)
                    self.logger.info(f"  Filter: {col_name} == {filter_value}")

                elif filter_name.endswith('_ne'):
                    # Not equal check: col_name_ne: value
                    col_name = filter_name.replace('_ne', '')
                    filter_mask = filter_mask & (pl.col(col_name) != filter_value)
                    self.logger.info(f"  Filter: {col_name} != {filter_value}")

                elif filter_name.endswith('_gt'):
                    # Greater than: col_name_gt: value
                    col_name = filter_name.replace('_gt', '')
                    filter_mask = filter_mask & (pl.col(col_name) > filter_value)
                    self.logger.info(f"  Filter: {col_name} > {filter_value}")

                elif filter_name.endswith('_lt'):
                    # Less than: col_name_lt: value
                    col_name = filter_name.replace('_lt', '')
                    filter_mask = filter_mask & (pl.col(col_name) < filter_value)
                    self.logger.info(f"  Filter: {col_name} < {filter_value}")

                else:
                    # Default: treat as equality
                    filter_mask = filter_mask & (pl.col(filter_name) == filter_value)
                    self.logger.info(f"  Filter: {filter_name} == {filter_value}")

            df_filtered = df.filter(filter_mask)
            n_filtered = len(df) - len(df_filtered)
            self.logger.info(f"Filtered out {n_filtered:,} objects")
            self.logger.info(f"Remaining: {len(df_filtered):,} objects ({len(df_filtered)/len(df)*100:.1f}%)")
            df = df_filtered

        # Check if we have the identifier column
        id_col = pred_config['data'].get('id_column', 'source_id')
        if id_col not in df.columns:
            self.logger.warning(f"ID column '{id_col}' not found, using row index")
            context['has_ids'] = False
        else:
            context['has_ids'] = True
            context['id_column'] = id_col

        context['prediction_data'] = df
        return context


class PreprocessPredictionDataStep(PipelineStep):
    """Preprocess prediction data (same as training)."""

    def __init__(self):
        super().__init__("Preprocess Prediction Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['prediction_data']
        pred_config = context['prediction_config']
        config = context['config']

        preprocessing = pred_config.get('preprocessing', {})

        # Add derived Gaia colors if missing
        if 'g' in df.columns and 'bp' in df.columns and 'rp' in df.columns:
            if 'g_bp' not in df.columns:
                self.logger.info("Computing derived color: g_bp = g - bp")
                df = df.with_columns([
                    (pl.col('g') - pl.col('bp')).alias('g_bp')
                ])
            if 'g_rp' not in df.columns:
                self.logger.info("Computing derived color: g_rp = g - rp")
                df = df.with_columns([
                    (pl.col('g') - pl.col('rp')).alias('g_rp')
                ])

        # Filter missing values if requested
        if preprocessing.get('filter_missing', True):
            missing_val = preprocessing.get('missing_value', config.get('processing', 'missing_value'))

            # Get required feature columns from model metadata
            required_features = context['model_metadata'].get('features', [])

            if required_features:
                # Build filter for required features
                mask = pl.lit(True)
                for feature in required_features:
                    # Only check features that exist in the data
                    # (some may be engineered later)
                    if feature in df.columns:
                        mask = mask & (pl.col(feature) != missing_val)

                df_clean = df.filter(mask)
                filtered = len(df) - len(df_clean)
                self.logger.info(f"Filtered {filtered:,} objects with missing values")
                self.logger.info(f"Remaining: {len(df_clean):,} objects ({len(df_clean)/len(df)*100:.1f}%)")
            else:
                self.logger.warning("No feature list in metadata, skipping missing value filter")
                df_clean = df
        else:
            df_clean = df

        context['clean_prediction_data'] = df_clean
        return context


class EngineerPredictionFeaturesStep(PipelineStep):
    """Engineer features for prediction (must match training)."""

    def __init__(self):
        super().__init__("Engineer Prediction Features")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['clean_prediction_data']
        pred_config = context['prediction_config']
        metadata = context['model_metadata']

        feature_config = pred_config.get('feature_engineering', {})

        if feature_config.get('enabled', False):
            self.logger.info("Feature engineering enabled")

            # Convert to pandas for feature engineering
            df_pd = df.to_pandas()

            color_cols = feature_config.get('color_cols', [])
            mag_cols = feature_config.get('mag_cols', [])

            self.logger.info(f"Color columns: {color_cols}")
            self.logger.info(f"Magnitude columns: {mag_cols}")

            # Engineer features (same function as training)
            df_features = engineer_all_features(
                df_pd,
                color_cols=color_cols,
                mag_cols=mag_cols
            )

            self.logger.info(f"Engineered features: {df_features.shape[1]} columns")

            # Convert back to polars
            df_result = pl.from_pandas(df_features)
        else:
            self.logger.info("Feature engineering disabled (using pre-engineered features)")
            df_result = df

        context['feature_prediction_data'] = df_result
        return context


class ApplyClusteringFeaturesStep(PipelineStep):
    """Apply clustering model to generate cluster probability features."""

    def __init__(self):
        super().__init__("Apply Clustering Features")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['feature_prediction_data']
        pred_config = context['prediction_config']
        config = context['config']
        metadata = context.get('model_metadata', {})

        # Check if clustering is enabled
        clustering_config = pred_config.get('clustering', {})
        if not clustering_config.get('enabled', False):
            self.logger.info("Clustering disabled, skipping")
            context['clustered_prediction_data'] = df
            return context

        self.logger.info("Applying clustering features...")

        # Get model ID from metadata or context
        model_id = metadata.get('model_id') or metadata.get('model_file', '').replace('.pkl', '')
        if not model_id:
            raise ValueError("Could not determine model ID for loading clustering artifacts")

        models_dir = Path(config.get_path('models'))

        # Load clustering model and scaler
        clustering_method = clustering_config.get('method', 'kmeans')
        clustering_model_file = models_dir / f"{model_id}_clustering_{clustering_method}.pkl"
        scaler_file = models_dir / f"{model_id}_clustering_scaler.pkl"

        if not clustering_model_file.exists():
            raise FileNotFoundError(f"Clustering model not found: {clustering_model_file}")
        if not scaler_file.exists():
            raise FileNotFoundError(f"Clustering scaler not found: {scaler_file}")

        self.logger.info(f"Loading clustering model: {clustering_model_file.name}")
        clustering_model = joblib.load(clustering_model_file)

        self.logger.info(f"Loading scaler: {scaler_file.name}")
        scaler = joblib.load(scaler_file)

        # Get feature columns for clustering (base photometry only)
        cluster_features = clustering_config.get('features', ['g', 'bp', 'rp', 'bp_rp', 'g_bp', 'g_rp'])
        self.logger.info(f"Using {len(cluster_features)} features for clustering: {cluster_features}")

        # Extract and scale features
        X = df.select(cluster_features).to_numpy().astype(np.float64)
        X_scaled = scaler.transform(X)

        # Generate cluster probabilities
        if hasattr(clustering_model, 'cluster_centers_'):
            # KMeans - calculate soft probabilities from distances
            distances = clustering_model.transform(X_scaled)
            inv_distances = 1.0 / (distances + 1e-10)
            probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)
            n_clusters = clustering_model.n_clusters
        elif hasattr(clustering_model, 'predict_proba'):
            # GMM - direct probability prediction
            probabilities = clustering_model.predict_proba(X_scaled)
            n_clusters = clustering_model.n_components
        else:
            raise ValueError(f"Unsupported clustering model type: {type(clustering_model)}")

        self.logger.info(f"Generated {n_clusters} cluster probability features")

        # Add cluster probabilities to dataframe
        cluster_cols = [f'cluster_prob_{i}' for i in range(n_clusters)]
        cluster_df = pl.DataFrame(probabilities, schema=cluster_cols)

        # Concatenate with original dataframe
        df_with_clusters = pl.concat([df, cluster_df], how='horizontal')

        self.logger.info(f"Added cluster features: {cluster_cols}")
        self.logger.info(f"  Probability mean: {probabilities.mean():.4f}")
        self.logger.info(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

        context['clustered_prediction_data'] = df_with_clusters
        context['cluster_cols'] = cluster_cols
        return context


class MakePredictionsStep(PipelineStep):
    """Make temperature predictions using trained model with uncertainty estimates."""

    def __init__(self):
        super().__init__("Make Predictions")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Use clustered data if available, otherwise use feature data
        df = context.get('clustered_prediction_data', context['feature_prediction_data'])
        model = context['model']
        metadata = context['model_metadata']
        pred_config = context['prediction_config']

        # Get feature columns from model metadata
        required_features = metadata.get('features', [])

        if not required_features:
            raise ValueError("Model metadata missing 'features' - cannot determine which columns to use")

        # Check which features are available
        available_features = [f for f in required_features if f in df.columns]
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features[:5]}... ({len(missing_features)} total)")

        self.logger.info(f"Using {len(available_features)} features for prediction")

        # Extract feature matrix (same column order as training)
        X_pred_df = df.select(available_features).to_pandas()
        # Use numpy to avoid "fitted without feature names" warning when model was fit with array
        X_pred = X_pred_df[available_features].to_numpy()

        # Make predictions
        self.logger.info(f"Predicting temperatures for {len(X_pred):,} objects...")
        predictions = model.predict(X_pred)

        # Calculate uncertainty estimates for Random Forest models
        uncertainties = None
        if hasattr(model, 'estimators_'):
            # Check if fast uncertainty estimation is requested
            uncertainty_config = pred_config.get('uncertainty', {})
            fast_mode = uncertainty_config.get('fast', False)
            n_sample_trees = uncertainty_config.get('n_sample_trees', 20)

            if fast_mode:
                self.logger.info(f"Calculating prediction uncertainties (FAST mode: sampling {n_sample_trees} trees)...")

                # Sample subset of trees for uncertainty estimation
                n_trees = len(model.estimators_)
                sample_indices = np.random.choice(
                    n_trees,
                    min(n_sample_trees, n_trees),
                    replace=False
                )

                # X_pred is already numpy (avoids feature name warnings)
                # Get predictions from sampled trees
                sampled_predictions = np.array([
                    model.estimators_[i].predict(X_pred) for i in sample_indices
                ])

                # Scale uncertainty to account for sampling
                uncertainties = np.std(sampled_predictions, axis=0) * np.sqrt(n_trees / n_sample_trees)

            else:
                self.logger.info("Calculating prediction uncertainties (FULL mode: all trees)...")

                # Get predictions from all individual trees (X_pred is numpy, avoids feature name warnings)
                tree_predictions = np.array([tree.predict(X_pred) for tree in model.estimators_])

                # Calculate standard deviation across trees as uncertainty
                uncertainties = np.std(tree_predictions, axis=0)

            # Calculate statistics
            unc_mean = np.mean(uncertainties)
            unc_median = np.median(uncertainties)
            unc_min = np.min(uncertainties)
            unc_max = np.max(uncertainties)

            mode_str = "FAST" if fast_mode else "FULL"
            self.logger.info(f"Uncertainty statistics ({mode_str}):")
            self.logger.info(f"  Mean:   {unc_mean:.3f}")
            self.logger.info(f"  Median: {unc_median:.3f}")
            self.logger.info(f"  Range:  [{unc_min:.3f}, {unc_max:.3f}]")

            context['uncertainties'] = uncertainties
            context['uncertainty_stats'] = {
                'mean': float(unc_mean),
                'median': float(unc_median),
                'min': float(unc_min),
                'max': float(unc_max),
                'fast_mode': fast_mode
            }
        else:
            self.logger.warning("Model does not support uncertainty estimation (not a Random Forest)")
            context['uncertainties'] = None
            context['uncertainty_stats'] = None

        # Calculate prediction statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)

        self.logger.info("Prediction statistics:")
        self.logger.info(f"  Mean: {pred_mean:.0f} K")
        self.logger.info(f"  Std:  {pred_std:.0f} K")
        self.logger.info(f"  Range: [{pred_min:.0f}, {pred_max:.0f}] K")

        context['predictions'] = predictions
        context['prediction_stats'] = {
            'mean': float(pred_mean),
            'std': float(pred_std),
            'min': float(pred_min),
            'max': float(pred_max),
            'n_objects': len(predictions)
        }

        return context


class ConvertLogPredictionsStep(PipelineStep):
    """Convert log-space predictions and uncertainties to Kelvin if model used log transform."""

    def __init__(self):
        super().__init__("Convert Log Predictions")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        predictions = context['predictions']
        uncertainties = context.get('uncertainties')
        pred_config = context['prediction_config']
        model_metadata = context.get('model_metadata', {})
        model_id = context.get('model_id', '')

        # Check if model uses log transformation
        # Look for indicators: target_transform in config, model_id contains '_log', or target contains 'log'
        uses_log = False

        # Check config
        target_transform = pred_config.get('target_transform', {})
        if target_transform.get('type') in ['log', 'log10']:
            uses_log = True
            self.logger.info("Detected log transformation from config")

        # Check model ID
        elif '_log' in model_id.lower() and 'logg' not in model_id.lower():
            uses_log = True
            self.logger.info("Detected log transformation from model ID")

        # Check metadata
        elif 'log' in model_metadata.get('target', '').lower() and 'logg' not in model_metadata.get('target', '').lower():
            uses_log = True
            self.logger.info("Detected log transformation from metadata")

        if not uses_log:
            self.logger.info("Model does not use log transformation, skipping conversion")
            return context

        # Convert predictions from log10(Teff) to Teff (Kelvin)
        self.logger.info("Converting predictions from log10(Teff) to Teff (Kelvin)...")
        teff_kelvin = 10 ** predictions

        # Convert uncertainties from log space to Kelvin
        # Formula: σ_Teff = Teff × σ_log(Teff) × ln(10)
        if uncertainties is not None:
            self.logger.info("Converting uncertainties from log space to Kelvin...")
            unc_kelvin = teff_kelvin * uncertainties * np.log(10)

            # Update uncertainty statistics
            unc_mean = np.mean(unc_kelvin)
            unc_median = np.median(unc_kelvin)
            unc_min = np.min(unc_kelvin)
            unc_max = np.max(unc_kelvin)

            self.logger.info(f"Converted uncertainty statistics:")
            self.logger.info(f"  Mean:   {unc_mean:.1f} K")
            self.logger.info(f"  Median: {unc_median:.1f} K")
            self.logger.info(f"  Range:  [{unc_min:.1f}, {unc_max:.1f}] K")

            context['uncertainties'] = unc_kelvin
            context['uncertainty_stats'] = {
                'mean': float(unc_mean),
                'median': float(unc_median),
                'min': float(unc_min),
                'max': float(unc_max),
                'fast_mode': context.get('uncertainty_stats', {}).get('fast_mode', False)
            }

        # Update predictions
        context['predictions'] = teff_kelvin

        # Update prediction statistics
        pred_mean = np.mean(teff_kelvin)
        pred_std = np.std(teff_kelvin)
        pred_min = np.min(teff_kelvin)
        pred_max = np.max(teff_kelvin)

        self.logger.info("Converted prediction statistics:")
        self.logger.info(f"  Mean: {pred_mean:.0f} K")
        self.logger.info(f"  Std:  {pred_std:.0f} K")
        self.logger.info(f"  Range: [{pred_min:.0f}, {pred_max:.0f}] K")

        context['prediction_stats'] = {
            'mean': float(pred_mean),
            'std': float(pred_std),
            'min': float(pred_min),
            'max': float(pred_max),
            'n_objects': len(teff_kelvin)
        }

        return context


class SavePredictionsStep(PipelineStep):
    """Save predictions to file."""

    def __init__(self):
        super().__init__("Save Predictions")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        predictions = context['predictions']
        uncertainties = context.get('uncertainties')
        pred_config = context['prediction_config']
        config = context['config']
        df = context['feature_prediction_data']
        model_id = context['model_id']

        # Determine output file
        output_config = pred_config.get('output', {})
        output_file = output_config.get('output_file', f'predictions_{model_id}.parquet')

        # Handle output directory
        if output_config.get('output_dir'):
            output_dir = Path(output_config['output_dir'])
        else:
            output_dir = Path(config.get_path('processed'))

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file

        # Determine target name from config, model metadata, or model filename
        target_name = pred_config.get('target', {}).get('name')

        if not target_name:
            # Try to get from model metadata
            metadata = context.get('model_metadata', {})
            target_name = metadata.get('target', metadata.get('target_name'))

        if not target_name:
            # Try to infer from model filename
            model_id = context.get('model_id', '')
            if 'logg' in model_id.lower():
                target_name = 'logg'
            elif 'teff' in model_id.lower():
                target_name = 'teff'
            elif 'feh' in model_id.lower() or 'mh' in model_id.lower():
                target_name = 'feh'
            else:
                # Default fallback
                target_name = 'teff'
                self.logger.warning(f"Could not detect target name, using default: {target_name}")

        # Normalize target name (strip common suffixes)
        if target_name:
            # Remove suffixes like _gaia, _predicted, _unified to get base parameter name
            for suffix in ['_gaia', '_predicted', '_unified', '_improved']:
                if target_name.endswith(suffix):
                    target_name = target_name.replace(suffix, '')
                    break

        # Build output dataframe
        result_data = {f'{target_name}_predicted': predictions}

        # Add uncertainty estimates if available
        if uncertainties is not None:
            result_data[f'{target_name}_uncertainty'] = uncertainties
            self.logger.info(f"Including uncertainty estimates in output")

        # Add IDs if available
        if context.get('has_ids'):
            id_col = context['id_column']
            result_data[id_col] = df[id_col].to_pandas()

        # Add input features if requested
        if output_config.get('include_features', False):
            feature_cols = context['model_metadata'].get('features', [])
            for col in feature_cols:
                if col in df.columns:
                    result_data[col] = df[col].to_pandas()

        # Add metadata columns if specified
        include_cols = output_config.get('include_columns', [])
        for col in include_cols:
            if col in df.columns:
                result_data[col] = df[col].to_pandas()

        # Create output dataframe
        result_df = pd.DataFrame(result_data)

        # Save predictions
        if output_path.suffix == '.parquet':
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            result_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        self.logger.info(f"✓ Saved predictions: {output_path.name}")
        self.logger.info(f"  Output columns: {list(result_df.columns)}")
        self.logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Save summary if requested
        if output_config.get('save_summary', True):
            summary_file = output_path.with_suffix('.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Prediction Summary\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Model: {model_id}\n")
                f.write(f"Model file: {context['model_file']}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write(f"Input data: {pred_config['data']['source_file']}\n")
                f.write(f"Objects: {context['prediction_stats']['n_objects']:,}\n\n")

                f.write("Prediction Statistics:\n")
                stats = context['prediction_stats']
                f.write(f"  Mean:  {stats['mean']:.1f} K\n")
                f.write(f"  Std:   {stats['std']:.1f} K\n")
                f.write(f"  Min:   {stats['min']:.1f} K\n")
                f.write(f"  Max:   {stats['max']:.1f} K\n\n")

                # Add uncertainty statistics if available
                if context.get('uncertainty_stats'):
                    f.write("Uncertainty Statistics:\n")
                    unc_stats = context['uncertainty_stats']
                    f.write(f"  Mean:   {unc_stats['mean']:.1f} K\n")
                    f.write(f"  Median: {unc_stats['median']:.1f} K\n")
                    f.write(f"  Min:    {unc_stats['min']:.1f} K\n")
                    f.write(f"  Max:    {unc_stats['max']:.1f} K\n\n")

                f.write(f"Output file: {output_path.name}\n")
                f.write(f"Output columns: {len(result_df.columns)}\n")
                if uncertainties is not None:
                    f.write(f"Includes uncertainty estimates: Yes\n")

            self.logger.info(f"✓ Saved summary: {summary_file.name}")

        context['output_file'] = str(output_path)
        context['output_summary'] = str(summary_file) if output_config.get('save_summary', True) else None

        return context


class PredictionPipeline(Pipeline):
    """
    Temperature prediction pipeline using trained models.

    This pipeline loads a trained model and applies it to new observations,
    ensuring consistent feature engineering between training and prediction.

    Steps:
    1. Load prediction configuration
    2. Load trained model + metadata
    3. Load prediction data
    4. Preprocess data (filter missing values)
    5. Engineer features (if needed)
    6. Make predictions
    7. Save results

    Usage
    -----
    >>> # Using config file
    >>> pipeline = PredictionPipeline('config/prediction/predict_all_sources.yaml')
    >>> context = pipeline.run()
    >>> print(f"Predictions saved: {context['output_file']}")

    >>> # Programmatic usage
    >>> from src.pipeline import PredictionPipeline
    >>> pipeline = PredictionPipeline('config/prediction/predict_all_sources.yaml')
    >>> context = pipeline.run()
    >>> predictions = context['predictions']  # numpy array

    Parameters
    ----------
    prediction_config_path : str
        Path to prediction configuration YAML file
    """

    def __init__(self, prediction_config_path: str):
        self.prediction_config_path = prediction_config_path

        steps = [
            LoadPredictionConfigStep(prediction_config_path),
            LoadTrainedModelStep(),
            LoadPredictionDataStep(),
            PreprocessPredictionDataStep(),
            EngineerPredictionFeaturesStep(),
            ApplyClusteringFeaturesStep(),
            MakePredictionsStep(),
            ConvertLogPredictionsStep(),  # Convert log predictions to Kelvin
            SavePredictionsStep(),
        ]

        # Extract name from config for pipeline name
        config_file = Path(prediction_config_path)
        pipeline_name = f"Prediction Pipeline ({config_file.stem})"

        super().__init__(pipeline_name, steps)

    def run(self) -> Dict[str, Any]:
        """Run the pipeline with initial context."""
        context = {
            'config': get_config(),
            'config_file': self.prediction_config_path
        }
        return super().run(context)
