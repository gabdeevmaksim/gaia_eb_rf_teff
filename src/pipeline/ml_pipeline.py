"""
Machine learning training pipeline for temperature prediction.

This pipeline orchestrates ML model training:
1. Load and prepare data
2. Engineer features
3. Select best features
4. Train model
5. Evaluate performance
6. Save model and artifacts
"""

import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import joblib
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep
from ..features import engineer_all_features, select_best_features, get_feature_importance


class LoadMLDataStep(PipelineStep):
    """Load ML training data from config (eb_unified_photometry in raw)."""

    def __init__(self):
        super().__init__("Load ML Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from ..config import get_config

        config = get_config()
        path = config.get_dataset_path("eb_unified_photometry", "raw")
        data = pd.read_parquet(path)
        self.logger.info(f"Loaded {len(data):,} objects")

        context['ml_data'] = data
        return context


class FeatureEngineeringStep(PipelineStep):
    """Engineer features for training."""

    def __init__(self, color_cols=None, mag_cols=None):
        super().__init__("Feature Engineering")
        self.color_cols = color_cols or ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp']
        self.mag_cols = mag_cols or ['gPSFMag']

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context['ml_data']

        # Filter for valid temperatures
        config = context['config']
        missing_val = config.get('processing', 'missing_value')
        data_clean = data[data['Te_avg'] != missing_val].copy()

        self.logger.info(f"Objects with valid temperatures: {len(data_clean):,}")

        # Engineer features
        data_features = engineer_all_features(
            data_clean,
            color_cols=self.color_cols,
            mag_cols=self.mag_cols
        )

        self.logger.info(f"Features created: {data_features.shape[1]} (from {data_clean.shape[1]})")

        context['data_features'] = data_features
        context['color_cols'] = self.color_cols
        context['mag_cols'] = self.mag_cols
        return context


class PrepareTrainTestStep(PipelineStep):
    """Prepare train/test split."""

    def __init__(self):
        super().__init__("Prepare Train/Test Split")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data_features = context['data_features']
        config = context['config']

        # Separate features and target
        exclude_cols = ['Te_avg', 'original_ext_source_id', 'Te_gr', 'Te_ri', 'Te_iz']
        feature_cols = [c for c in data_features.columns if c not in exclude_cols]

        X = data_features[feature_cols]
        y = data_features['Te_avg']

        self.logger.info(f"Features: {len(feature_cols)}")
        self.logger.info(f"Target: Te_avg")

        # Train/test split
        test_size = config.get('ml', 'test_size')
        random_state = config.get('ml', 'random_state')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        self.logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        context['X_train'] = X_train
        context['X_test'] = X_test
        context['y_train'] = y_train
        context['y_test'] = y_test
        context['feature_cols'] = feature_cols

        return context


class TrainModelStep(PipelineStep):
    """Train Random Forest model."""

    def __init__(self, n_estimators=None, max_depth=None):
        super().__init__("Train Model")
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context['X_train']
        y_train = context['y_train']
        config = context['config']

        # Get hyperparameters
        n_estimators = self.n_estimators or config.get('ml', 'rf_n_estimators')
        max_depth = self.max_depth or config.get('ml', 'rf_max_depth')
        min_samples_split = config.get('ml', 'rf_min_samples_split')
        min_samples_leaf = config.get('ml', 'rf_min_samples_leaf')
        max_features = config.get('ml', 'rf_max_features')
        random_state = config.get('ml', 'random_state')
        n_jobs = config.get('ml', 'rf_n_jobs')

        self.logger.info(f"Training Random Forest:")
        self.logger.info(f"  n_estimators: {n_estimators}")
        self.logger.info(f"  max_depth: {max_depth}")
        self.logger.info(f"  n_jobs: {n_jobs}")

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )

        model.fit(X_train, y_train)

        self.logger.info("✓ Model trained")

        context['model'] = model
        return context


class EvaluateModelStep(PipelineStep):
    """Evaluate model performance."""

    def __init__(self):
        super().__init__("Evaluate Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        X_test = context['X_test']
        y_test = context['y_test']

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Relative error
        relative_errors = np.abs((y_test - y_pred) / y_test)
        mean_rel_error = np.mean(relative_errors) * 100

        # Percentage within thresholds
        within_5 = (relative_errors <= 0.05).sum() / len(y_test) * 100
        within_10 = (relative_errors <= 0.10).sum() / len(y_test) * 100
        within_20 = (relative_errors <= 0.20).sum() / len(y_test) * 100

        self.logger.info("Performance Metrics:")
        self.logger.info(f"  MAE:  {mae:.0f} K")
        self.logger.info(f"  RMSE: {rmse:.0f} K")
        self.logger.info(f"  R²:   {r2:.4f}")
        self.logger.info(f"  Mean Relative Error: {mean_rel_error:.2f}%")
        self.logger.info(f"  Within 5%:  {within_5:.1f}%")
        self.logger.info(f"  Within 10%: {within_10:.1f}%")
        self.logger.info(f"  Within 20%: {within_20:.1f}%")

        context['y_pred'] = y_pred
        context['metrics'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_relative_error': float(mean_rel_error),
            'within_5_percent': float(within_5),
            'within_10_percent': float(within_10),
            'within_20_percent': float(within_20)
        }

        return context


class SaveModelStep(PipelineStep):
    """Save trained model and artifacts."""

    def __init__(self):
        super().__init__("Save Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        metrics = context['metrics']
        feature_cols = context['feature_cols']
        config = context['config']

        # Create model ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"rf_temperature_regressor_{timestamp}"

        models_dir = config.get_path('models', ensure_exists=True)

        # Save model
        model_file = models_dir / f"{model_id}.pkl"
        joblib.dump(model, model_file)
        self.logger.info(f"Saved model: {model_file.name}")

        # Save metadata
        metadata = {
            'model_id': model_id,
            'timestamp': timestamp,
            'model_type': 'RandomForestRegressor',
            'n_features': len(feature_cols),
            'features': feature_cols,
            'metrics': metrics,
            'hyperparameters': model.get_params()
        }

        metadata_file = models_dir / f"{model_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata: {metadata_file.name}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'y_true': context['y_test'],
            'y_pred': context['y_pred']
        })
        pred_file = models_dir / f"{model_id}_test_predictions.parquet"
        predictions_df.to_parquet(pred_file)
        self.logger.info(f"Saved predictions: {pred_file.name}")

        context['model_id'] = model_id
        context['model_file'] = model_file
        context['metadata_file'] = metadata_file

        return context


class MLTrainingPipeline(Pipeline):
    """
    Complete ML training pipeline.

    Steps:
    1. Load ML data
    2. Engineer features
    3. Prepare train/test split
    4. Train model
    5. Evaluate performance
    6. Save model and artifacts

    Usage
    -----
    >>> pipeline = MLTrainingPipeline()
    >>> context = pipeline.run()
    >>> model_id = context['model_id']
    >>> print(f"Model saved: {model_id}")
    """

    def __init__(self, n_estimators=None, max_depth=None):
        steps = [
            LoadMLDataStep(),
            FeatureEngineeringStep(),
            PrepareTrainTestStep(),
            TrainModelStep(n_estimators=n_estimators, max_depth=max_depth),
            EvaluateModelStep(),
            SaveModelStep(),
        ]

        super().__init__("ML Training Pipeline", steps)
