"""
Validation pipeline for model evaluation and visualization.

This pipeline generates standardized validation plots and metrics for trained models,
making it easy to compare model performance and identify issues.

Usage:
    # Using config file
    pipeline = ValidationPipeline('config/validation/validate_gaia_2mass_ir.yaml')
    context = pipeline.run()

    # Or from CLI
    python pipeline.py validate --config config/validation/validate_gaia_2mass_ir.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import yaml
import json
import joblib

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep
from ..config import get_config
from ..visualization import validation_plots


class LoadValidationConfigStep(PipelineStep):
    """Load validation configuration from YAML file."""

    def __init__(self, config_path: str):
        super().__init__("Load Validation Configuration")
        self.config_path = config_path

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Validation config not found: {config_file}")

        with open(config_file, 'r') as f:
            val_config = yaml.safe_load(f)

        self.logger.info(f"Loaded validation config: {config_file.name}")
        self.logger.info(f"Model pattern: {val_config['model']['model_pattern']}")

        context['validation_config'] = val_config
        return context


class LoadModelForValidationStep(PipelineStep):
    """Load trained model, metadata, and test predictions."""

    def __init__(self):
        super().__init__("Load Model for Validation")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        val_config = context['validation_config']
        config = context['config']

        model_pattern = val_config['model']['model_pattern']
        models_dir = Path(config.get_path('models'))

        # Find model files
        if '*' in model_pattern:
            model_files = sorted(models_dir.glob(f"{model_pattern}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model found matching: {model_pattern}")
            model_file = model_files[-1]  # Most recent
            self.logger.info(f"Found {len(model_files)} matching models, using most recent")
        else:
            model_file = models_dir / f"{model_pattern}.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")

        model_id = model_file.stem
        self.logger.info(f"Validating model: {model_id}")

        # Load model
        model = joblib.load(model_file)

        # Load metadata
        metadata_file = model_file.with_name(f"{model_id}_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"  Model: {metadata.get('model_name', 'Unknown')}")
            self.logger.info(f"  Features: {metadata.get('n_features', 'Unknown')}")
        else:
            self.logger.warning("Metadata file not found")
            metadata = {'model_name': model_id}

        # Load test predictions
        pred_file = model_file.with_name(f"{model_id}_test_predictions.parquet")
        if pred_file.exists():
            test_pred = pd.read_parquet(pred_file)
            self.logger.info(f"  Test predictions: {len(test_pred):,} samples")
        else:
            raise FileNotFoundError(f"Test predictions not found: {pred_file.name}")

        context['model'] = model
        context['model_metadata'] = metadata
        context['model_id'] = model_id
        context['model_file'] = str(model_file)
        context['test_predictions'] = test_pred

        return context


class LoadTrainingDataStep(PipelineStep):
    """Load training data with color information for color-temperature plots (optional)."""

    def __init__(self):
        super().__init__("Load Training Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        val_config = context['validation_config']
        config = context['config']
        metadata = context['model_metadata']

        # Only load if color-temperature plots are requested
        if not val_config.get('plots', {}).get('color_temp_relations', False):
            self.logger.info("Color-temperature plots not requested, skipping training data load")
            context['training_data_with_colors'] = None
            return context

        # Try to get data source from metadata
        data_config = metadata.get('data_config')
        if not data_config:
            self.logger.warning("No data config in metadata, cannot load training data")
            context['training_data_with_colors'] = None
            return context

        source_file = data_config.get('source_file')
        source_location = data_config.get('source_location', 'processed')

        if not source_file:
            self.logger.warning("No source file in metadata, cannot load training data")
            context['training_data_with_colors'] = None
            return context

        # Load data
        import polars as pl
        from astropy.table import Table

        data_dir = Path(config.get_path(source_location))
        data_path = data_dir / source_file

        if not data_path.exists():
            self.logger.warning(f"Training data file not found: {data_path}")
            context['training_data_with_colors'] = None
            return context

        self.logger.info(f"Loading training data from {source_location}/{source_file}")

        # Load based on format
        if data_path.suffix == '.parquet':
            df_pl = pl.read_parquet(data_path)
            df = df_pl.to_pandas()
        elif data_path.suffix == '.csv':
            df_pl = pl.read_csv(data_path)
            df = df_pl.to_pandas()
        elif data_path.suffix == '.fits':
            table = Table.read(str(data_path))
            df = table.to_pandas()
        else:
            self.logger.warning(f"Unsupported file format: {data_path.suffix}")
            context['training_data_with_colors'] = None
            return context

        # Set source_id as index if available
        id_column = data_config.get('id_column', 'source_id')
        if id_column in df.columns:
            df = df.set_index(id_column)

        self.logger.info(f"Loaded {len(df):,} samples with {df.shape[1]} columns")
        context['training_data_with_colors'] = df

        return context


class CalculateMetricsStep(PipelineStep):
    """Calculate comprehensive validation metrics."""

    def __init__(self):
        super().__init__("Calculate Metrics")

    def _get_target_info(self, target_col: str) -> Dict[str, str]:
        """Get display information for target variable."""
        target_map = {
            'teff_gaia': {'name': 'Temperature', 'unit': 'K', 'short': 'Teff'},
            'logg_gaia': {'name': 'Surface Gravity', 'unit': 'dex', 'short': 'logg'},
            'feh_gaia': {'name': 'Metallicity', 'unit': 'dex', 'short': '[Fe/H]'},
            'mh_gaia': {'name': 'Metallicity', 'unit': 'dex', 'short': '[M/H]'},
        }

        # Check for exact match
        if target_col in target_map:
            return target_map[target_col]

        # Check for partial matches (case-insensitive)
        target_lower = target_col.lower()
        if 'teff' in target_lower or 'temp' in target_lower:
            return target_map['teff_gaia']
        elif 'logg' in target_lower:
            return target_map['logg_gaia']
        elif 'feh' in target_lower or 'fe_h' in target_lower:
            return target_map['feh_gaia']
        elif 'mh' in target_lower or 'm_h' in target_lower:
            return target_map['mh_gaia']

        # Default fallback
        return {'name': 'Value', 'unit': '', 'short': 'value'}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        test_pred = context['test_predictions']
        metadata = context.get('model_metadata', context.get('metadata', {}))

        # Get target information from metadata
        target_col = metadata.get('target', 'teff_gaia')
        target_info = self._get_target_info(target_col)
        context['target_info'] = target_info

        # Rename columns if needed for consistency - use generic names
        if 'y_true' in test_pred.columns and 'y_pred' in test_pred.columns:
            test_pred = test_pred.rename(columns={
                'y_true': 'true_value',
                'y_pred': 'predicted_value'
            })
        elif 'teff_true' in test_pred.columns and 'teff_pred' in test_pred.columns:
            test_pred = test_pred.rename(columns={
                'teff_true': 'true_value',
                'teff_pred': 'predicted_value'
            })

        y_true = test_pred['true_value']
        y_pred = test_pred['predicted_value']

        # Add residual column for plots
        test_pred['residual'] = y_pred - y_true

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Relative errors
        relative_errors = np.abs((y_true - y_pred) / y_true)
        mean_rel_error = np.mean(relative_errors) * 100

        # Percentage within thresholds
        within_5 = (relative_errors <= 0.05).sum() / len(y_true) * 100
        within_10 = (relative_errors <= 0.10).sum() / len(y_true) * 100
        within_20 = (relative_errors <= 0.20).sum() / len(y_true) * 100

        # Value bin statistics (by target value)
        bin_stats = self._calculate_performance_by_bins(y_true, y_pred, n_bins=5, target_info=target_info)

        # Format output with appropriate units
        unit = target_info['unit']
        unit_str = f" {unit}" if unit else ""

        self.logger.info("Validation Metrics:")
        self.logger.info(f"  MAE:  {mae:.4f}{unit_str}")
        self.logger.info(f"  RMSE: {rmse:.4f}{unit_str}")
        self.logger.info(f"  R²:   {r2:.4f}")
        self.logger.info(f"  Mean Relative Error: {mean_rel_error:.2f}%")
        self.logger.info(f"  Within 5%:  {within_5:.1f}%")
        self.logger.info(f"  Within 10%: {within_10:.1f}%")
        self.logger.info(f"  Within 20%: {within_20:.1f}%")

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_relative_error': float(mean_rel_error),
            'within_5_percent': float(within_5),
            'within_10_percent': float(within_10),
            'within_20_percent': float(within_20),
            'n_samples': len(y_true),
            'value_bins': bin_stats
        }

        context['metrics'] = metrics
        context['test_predictions'] = test_pred  # With renamed columns
        return context

    def _calculate_performance_by_bins(self, y_true, y_pred, n_bins=5, target_info=None):
        """Calculate metrics by value bins using fixed round boundaries.

        Bins are chosen based on the target variable type to be physically meaningful.
        """
        # Determine bins based on target variable
        if target_info and 'logg' in target_info.get('short', '').lower():
            # Surface gravity bins (dex)
            # Giants (0-2.5), Subgiants (2.5-3.5), Dwarfs (3.5-4.5), High gravity (4.5-6)
            bins = np.array([0, 2.5, 3.5, 4.5, 6.0])
        elif target_info and ('feh' in target_info.get('short', '').lower() or
                              'm/h' in target_info.get('short', '').lower() or
                              'mh' in target_info.get('short', '').lower()):
            # Metallicity bins (dex)
            # Metal-poor (<-1), Sub-solar (-1 to -0.5), Solar (-0.5 to 0.5), Metal-rich (>0.5)
            bins = np.array([-3.0, -1.0, -0.5, 0.5, 2.0])
        else:
            # Temperature bins (K) - default
            # Based on stellar classification (cool/solar/hot stars)
            bins = np.array([0, 4000, 5000, 6000, 8000, 50000])

        bin_stats = []
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if mask.sum() == 0:
                continue

            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]

            mae_bin = mean_absolute_error(y_true_bin, y_pred_bin)
            rmse_bin = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
            r2_bin = r2_score(y_true_bin, y_pred_bin)

            # Calculate percentage metrics
            relative_errors_bin = np.abs((y_true_bin - y_pred_bin) / y_true_bin)
            mean_pct_bin = np.mean(relative_errors_bin) * 100
            within_10_bin = (relative_errors_bin <= 0.10).sum() / len(y_true_bin) * 100

            bin_stats.append({
                'value_min': float(bins[i]),
                'value_max': float(bins[i+1]),
                'value_center': float((bins[i] + bins[i+1]) / 2),
                'n_samples': int(mask.sum()),
                'mae': float(mae_bin),
                'rmse': float(rmse_bin),
                'r2': float(r2_bin),
                'mean_pct': float(mean_pct_bin),
                'within_10': float(within_10_bin)
            })

        return bin_stats


class GenerateValidationPlotsStep(PipelineStep):
    """Generate standard validation plots."""

    def __init__(self):
        super().__init__("Generate Validation Plots")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        val_config = context['validation_config']
        test_pred = context['test_predictions']
        metrics = context['metrics']
        model_id = context['model_id']
        metadata = context['model_metadata']
        target_info = context['target_info']

        # Determine output directory
        output_config = val_config.get('output', {})
        figures_subdir = output_config.get('figures_subdir', f'{model_id}_validation')

        model_name = metadata.get('model_name', model_id)

        # Get requested plots
        requested_plots = val_config.get('plots', {})

        plot_count = 0

        # 1. Test scatter plot
        if requested_plots.get('test_scatter', True):
            self.logger.info("Generating test scatter plot...")
            validation_plots.plot_test_scatter(
                test_pred=test_pred,
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                r2=metrics['r2'],
                model_id=model_id,
                subdir=figures_subdir,
                model_name=model_name,
                target_info=target_info
            )
            plot_count += 1

        # 2. Residuals plot
        if requested_plots.get('residuals', True):
            self.logger.info("Generating residuals plot...")
            validation_plots.plot_residuals(
                test_pred=test_pred,
                model_id=model_id,
                subdir=figures_subdir,
                target_info=target_info
            )
            plot_count += 1

        # 3. Performance by value bins
        if requested_plots.get('performance_by_temp', True):
            self.logger.info(f"Generating performance by {target_info['short']} plot...")
            # Convert bin_stats to DataFrame and add 'bin' column for labels
            bin_stats_df = pd.DataFrame(metrics['value_bins'])
            unit = target_info['unit']

            # Create clean bin labels based on target variable type
            bin_labels = []
            target_short = target_info.get('short', '').lower()

            for _, row in bin_stats_df.iterrows():
                v_min = row['value_min']
                v_max = row['value_max']

                # Determine if we should use integers or decimals
                if 'logg' in target_short or 'feh' in target_short or 'mh' in target_short:
                    # Use 1 decimal for logg and metallicity
                    v_min_str = f"{v_min:.1f}"
                    v_max_str = f"{v_max:.1f}"
                else:
                    # Use integers for temperature
                    v_min_str = f"{int(v_min)}"
                    v_max_str = f"{int(v_max)}"

                # Create base label
                if 'logg' in target_short:
                    # logg bins
                    if v_min == 0:
                        label = f"<{v_max_str} {unit}"
                    elif v_max >= 6.0:
                        label = f">{v_min_str} {unit}"
                    else:
                        label = f"{v_min_str}-{v_max_str} {unit}"

                    # Add descriptive labels
                    if v_min == 0 and v_max == 2.5:
                        label += " (Giants)"
                    elif v_min == 2.5 and v_max == 3.5:
                        label += " (Subgiants)"
                    elif v_min == 3.5 and v_max == 4.5:
                        label += " (Dwarfs)"
                    elif v_min == 4.5:
                        label += " (High-g)"

                elif 'feh' in target_short or 'mh' in target_short:
                    # Metallicity bins
                    if v_min <= -3.0:
                        label = f"<{v_max_str} {unit}"
                    elif v_max >= 2.0:
                        label = f">{v_min_str} {unit}"
                    else:
                        label = f"{v_min_str} to {v_max_str} {unit}"

                    # Add descriptive labels
                    if v_max == -1.0:
                        label += " (Metal-poor)"
                    elif v_min == -1.0 and v_max == -0.5:
                        label += " (Sub-solar)"
                    elif v_min == -0.5 and v_max == 0.5:
                        label += " (Solar)"
                    elif v_min == 0.5:
                        label += " (Metal-rich)"

                else:
                    # Temperature bins (default)
                    if int(v_min) == 0:
                        label = f"<{v_max_str} {unit}"
                    elif int(v_max) >= 50000:
                        label = f">{v_min_str} {unit}"
                    else:
                        label = f"{v_min_str}-{v_max_str} {unit}"

                    # Add descriptive labels for temperature
                    if int(v_min) == 0 and int(v_max) == 4000:
                        label += " (Cool)"
                    elif int(v_min) == 4000 and int(v_max) == 5000:
                        label += " (Solar)"
                    elif int(v_max) >= 50000:
                        label += " (Hot)"

                bin_labels.append(label)

            bin_stats_df['bin'] = bin_labels
            validation_plots.plot_performance_by_temp(
                test_pred=test_pred,
                bin_stats_df=bin_stats_df,
                model_id=model_id,
                subdir=figures_subdir,
                target_info=target_info
            )
            plot_count += 1

        # 4. Value distributions
        if requested_plots.get('temp_distributions', True):
            self.logger.info(f"Generating {target_info['short']} distributions plot...")
            # For validation, we compare true_value vs predicted_value
            train_for_plot = test_pred[['true_value']].rename(columns={'true_value': 'teff_gspphot'})
            pred_for_plot = test_pred[['predicted_value']].rename(columns={'predicted_value': 'teff_predicted'})
            validation_plots.plot_temp_distributions(
                train_data=train_for_plot,
                predictions=pred_for_plot,
                model_id=model_id,
                subdir=figures_subdir,
                model_name=model_name,
                target_info=target_info
            )
            plot_count += 1

        # 5. Feature importance (if available)
        if requested_plots.get('feature_importance', True):
            model = context['model']
            if hasattr(model, 'feature_importances_'):
                self.logger.info("Generating feature importance plot...")
                feature_names = metadata.get('features', [f'Feature {i}' for i in range(len(model.feature_importances_))])
                validation_plots.plot_feature_importance(
                    feature_importance=dict(zip(feature_names, model.feature_importances_)),
                    model_id=model_id,
                    subdir=figures_subdir,
                    top_n=15
                )
                plot_count += 1

        # 6. Color-parameter relations (if requested and color data available in test_pred)
        if requested_plots.get('color_temp_relations', False):
            self.logger.info(f"Generating color-{target_info['short']} relation plots...")

            # Get color columns to plot from config
            color_configs = val_config.get('plots', {}).get('color_columns', [])

            if not color_configs:
                self.logger.warning("No color columns specified in validation config")
            else:
                # Check if color columns are in test predictions
                for color_cfg in color_configs:
                    color_col = color_cfg.get('column')
                    color_label = color_cfg.get('label', color_col)

                    if color_col and color_col in test_pred.columns:
                        # Prepare train and prediction data for plotting
                        # Use test set true values as "training" data for color-param plot
                        train_for_plot = test_pred[['true_value', color_col]].copy()
                        train_for_plot = train_for_plot.rename(columns={'true_value': 'teff_gspphot'})

                        pred_for_plot = test_pred[['predicted_value', color_col]].copy()
                        pred_for_plot = pred_for_plot.rename(columns={'predicted_value': 'teff_predicted'})

                        validation_plots.plot_color_temp_relations(
                            train_data=train_for_plot,
                            predictions=pred_for_plot,
                            color_col=color_col,
                            color_label=color_label,
                            model_id=model_id,
                            subdir=figures_subdir,
                            train_temp_col='teff_gspphot',
                            pred_temp_col='teff_predicted',
                            target_info=target_info
                        )
                        plot_count += 1
                    else:
                        self.logger.warning(f"Color column '{color_col}' not found in test predictions")

        self.logger.info(f"✓ Generated {plot_count} validation plots")
        context['n_plots_generated'] = plot_count
        context['figures_subdir'] = figures_subdir

        return context


class SaveValidationReportStep(PipelineStep):
    """Save comprehensive validation report."""

    def __init__(self):
        super().__init__("Save Validation Report")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        val_config = context['validation_config']
        metrics = context['metrics']
        model_id = context['model_id']
        metadata = context['model_metadata']
        config = context['config']
        target_info = context['target_info']

        output_config = val_config.get('output', {})

        # Determine report file location
        if output_config.get('report_file'):
            report_file = Path(output_config['report_file'])
        else:
            reports_dir = Path(config.get_path('reports'))
            report_file = reports_dir / f'validation_report_{model_id}.txt'

        report_file.parent.mkdir(parents=True, exist_ok=True)

        # Format units
        unit = target_info['unit']
        unit_str = f" {unit}" if unit else ""

        # Write report
        with open(report_file, 'w') as f:
            f.write(f"Validation Report: {metadata.get('model_name', model_id)}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Model ID: {model_id}\n")
            f.write(f"Model File: {context['model_file']}\n")
            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Test Set: {metrics['n_samples']:,} samples\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  MAE:  {metrics['mae']:.4f}{unit_str}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}{unit_str}\n")
            f.write(f"  R²:   {metrics['r2']:.4f}\n")
            f.write(f"  Mean Relative Error: {metrics['mean_relative_error']:.2f}%\n\n")

            f.write("ACCURACY THRESHOLDS:\n")
            f.write(f"  Within  5%: {metrics['within_5_percent']:.1f}%\n")
            f.write(f"  Within 10%: {metrics['within_10_percent']:.1f}%\n")
            f.write(f"  Within 20%: {metrics['within_20_percent']:.1f}%\n\n")

            f.write(f"PERFORMANCE BY {target_info['name'].upper()} RANGE:\n")
            for bin_stat in metrics['value_bins']:
                f.write(f"\n  [{bin_stat['value_min']:.2f} - {bin_stat['value_max']:.2f}]{unit_str} "
                       f"(n={bin_stat['n_samples']:,}):\n")
                f.write(f"    MAE:  {bin_stat['mae']:.4f}{unit_str}\n")
                f.write(f"    RMSE: {bin_stat['rmse']:.4f}{unit_str}\n")
                f.write(f"    R²:   {bin_stat['r2']:.4f}\n")

            f.write(f"\nVALIDATION PLOTS:\n")
            f.write(f"  Location: reports/figures/{context['figures_subdir']}/\n")
            f.write(f"  Number of plots: {context['n_plots_generated']}\n")

        self.logger.info(f"✓ Saved validation report: {report_file.name}")

        # Save metrics as JSON for programmatic access
        metrics_file = report_file.with_suffix('.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'model_id': model_id,
                'model_name': metadata.get('model_name', model_id),
                'validation_date': datetime.now().isoformat(),
                'metrics': metrics
            }, f, indent=2)

        self.logger.info(f"✓ Saved metrics JSON: {metrics_file.name}")

        context['report_file'] = str(report_file)
        context['metrics_file'] = str(metrics_file)

        return context


class ValidationPipeline(Pipeline):
    """
    Model validation pipeline for generating plots and metrics.

    This pipeline takes a trained model and generates comprehensive validation
    visualizations and performance metrics, making it easy to assess model quality
    and compare different models.

    Steps:
    1. Load validation configuration
    2. Load model + metadata + test predictions
    3. Calculate metrics (MAE, RMSE, R², etc.)
    4. Generate validation plots (scatter, residuals, etc.)
    5. Save validation report

    Usage
    -----
    >>> # Using config file
    >>> pipeline = ValidationPipeline('config/validation/validate_gaia_2mass_ir.yaml')
    >>> context = pipeline.run()
    >>> print(f"Report saved: {context['report_file']}")

    >>> # Programmatic usage
    >>> from src.pipeline import ValidationPipeline
    >>> pipeline = ValidationPipeline('config/validation/validate_model.yaml')
    >>> context = pipeline.run()
    >>> metrics = context['metrics']

    Parameters
    ----------
    validation_config_path : str
        Path to validation configuration YAML file
    """

    def __init__(self, validation_config_path: str):
        self.validation_config_path = validation_config_path

        steps = [
            LoadValidationConfigStep(validation_config_path),
            LoadModelForValidationStep(),
            CalculateMetricsStep(),
            GenerateValidationPlotsStep(),
            SaveValidationReportStep(),
        ]

        # Extract name from config for pipeline name
        config_file = Path(validation_config_path)
        pipeline_name = f"Validation Pipeline ({config_file.stem})"

        super().__init__(pipeline_name, steps)

    def run(self) -> Dict[str, Any]:
        """Run the pipeline with initial context."""
        context = {
            'config': get_config(),
            'config_file': self.validation_config_path
        }
        return super().run(context)
