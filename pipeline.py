#!/usr/bin/env python3
"""
Master pipeline orchestrator for eclipsing binary temperature analysis.

This script provides a command-line interface to run different pipelines:
- Data processing pipeline
- ML training pipeline
- Complete end-to-end pipeline

Usage:
    # Run complete pipeline
    python pipeline.py --all

    # Run data processing only
    python pipeline.py --data

    # Run ML training only
    python pipeline.py --ml

    # Dry run (show what would be executed)
    python pipeline.py --all --dry-run

    # Custom model parameters
    python pipeline.py --ml --n-estimators 500 --max-depth 25
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Import pipelines
from src.pipeline import (
    DataProcessingPipeline,
    MLTrainingPipeline,
    ConfigurableMLPipeline,
    PredictionPipeline,
    ValidationPipeline
)
from src.config import get_config


def run_data_pipeline(dry_run=False):
    """Run the data processing pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute Data Processing Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Convert ECSV to Parquet")
        logger.info("    2. Extract Pan-STARRS Duplicates")
        logger.info("    3. Merge Duplicates")
        logger.info("    4. Clean Photometry")
        logger.info("    5. Calculate Temperatures")
        return {}

    pipeline = DataProcessingPipeline()
    context = pipeline.run()
    return context


def run_ml_pipeline(n_estimators=None, max_depth=None, model_config=None, dry_run=False):
    """Run the ML training pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute ML Training Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Load ML Data")
        logger.info("    2. Engineer Features")
        logger.info("    3. Prepare Train/Test Split")
        logger.info("    4. Train Model")
        logger.info("    5. Evaluate Performance")
        logger.info("    6. Save Model")
        if model_config:
            logger.info(f"  Model config: {model_config}")
        elif n_estimators:
            logger.info(f"  Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        return {}

    # Use configurable pipeline if model config provided
    if model_config:
        logger.info(f"Using configurable pipeline with config: {model_config}")
        pipeline = ConfigurableMLPipeline(model_config)
        context = pipeline.run()
    else:
        logger.info("Using legacy pipeline (consider migrating to configurable pipeline)")
        pipeline = MLTrainingPipeline(n_estimators=n_estimators, max_depth=max_depth)
        context = pipeline.run()

    return context


def run_prediction_pipeline(pred_config=None, dry_run=False):
    """Run the prediction pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute Prediction Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Load Prediction Configuration")
        logger.info("    2. Load Trained Model")
        logger.info("    3. Load Prediction Data")
        logger.info("    4. Preprocess Data")
        logger.info("    5. Engineer Features")
        logger.info("    6. Make Predictions")
        logger.info("    7. Save Results")
        if pred_config:
            logger.info(f"  Prediction config: {pred_config}")
        return {}

    if not pred_config:
        raise ValueError("Prediction config file required (use --pred-config)")

    logger.info(f"Using prediction pipeline with config: {pred_config}")
    pipeline = PredictionPipeline(pred_config)
    context = pipeline.run()

    return context


def run_validation_pipeline(val_config=None, dry_run=False):
    """Run the validation pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute Validation Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Load Validation Configuration")
        logger.info("    2. Load Model + Metadata + Test Predictions")
        logger.info("    3. Calculate Metrics (MAE, RMSE, R², etc.)")
        logger.info("    4. Generate Validation Plots")
        logger.info("    5. Save Validation Report")
        if val_config:
            logger.info(f"  Validation config: {val_config}")
        return {}

    if not val_config:
        raise ValueError("Validation config file required (use --val-config)")

    logger.info(f"Using validation pipeline with config: {val_config}")
    pipeline = ValidationPipeline(val_config)
    context = pipeline.run()

    return context


def run_complete_pipeline(n_estimators=None, max_depth=None, model_config=None, dry_run=False):
    """Run complete end-to-end pipeline."""
    logger.info("=" * 70)
    logger.info("COMPLETE PIPELINE: Data Processing + ML Training")
    logger.info("=" * 70)
    logger.info("")

    # Run data processing
    logger.info("STAGE 1: Data Processing")
    logger.info("-" * 70)
    data_context = run_data_pipeline(dry_run=dry_run)

    if not dry_run:
        logger.info("")
        logger.info("Data processing complete!")
        logger.info("")

    # Run ML training
    logger.info("STAGE 2: ML Training")
    logger.info("-" * 70)
    ml_context = run_ml_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        model_config=model_config,
        dry_run=dry_run
    )

    if not dry_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("COMPLETE PIPELINE FINISHED")
        logger.info("=" * 70)

        if 'model_id' in ml_context:
            logger.info(f"✓ Model saved: {ml_context['model_id']}")
            # Support both old and new metrics format
            metrics = ml_context.get('metrics') or ml_context.get('test_metrics', {})
            if metrics:
                logger.info(f"  MAE: {metrics['mae']:.0f} K")
                logger.info(f"  R²: {metrics['r2']:.4f}")

    return {**data_context, **ml_context}


def main():
    parser = argparse.ArgumentParser(
        description='Eclipsing Binary Temperature Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py --all

  # Run data processing only
  python pipeline.py --data

  # Run ML training with model config (RECOMMENDED)
  python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml
  python pipeline.py --ml --ml-config config/models/panstarrs_unified.yaml

  # Run prediction with trained model
  python pipeline.py --predict --pred-config config/prediction/predict_gaia_2mass_ir.yaml
  python pipeline.py --predict --pred-config config/prediction/predict_panstarrs_unified.yaml

  # Run validation (generate plots + metrics)
  python pipeline.py --validate --val-config config/validation/validate_gaia_2mass_ir.yaml
  python pipeline.py --validate --val-config config/validation/validate_latest_model.yaml

  # Legacy: Custom model parameters (use --ml-config instead)
  python pipeline.py --ml --n-estimators 500 --max-depth 25

  # Dry run (see what would be executed)
  python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml --dry-run
  python pipeline.py --predict --pred-config config/prediction/predict_all.yaml --dry-run
  python pipeline.py --validate --val-config config/validation/validate_gaia_2mass_ir.yaml --dry-run
        """
    )

    # Pipeline selection
    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (data processing + ML training)'
    )
    pipeline_group.add_argument(
        '--data',
        action='store_true',
        help='Run data processing pipeline only'
    )
    pipeline_group.add_argument(
        '--ml',
        action='store_true',
        help='Run ML training pipeline only'
    )
    pipeline_group.add_argument(
        '--predict',
        action='store_true',
        help='Run prediction pipeline only'
    )
    pipeline_group.add_argument(
        '--validate',
        action='store_true',
        help='Run validation pipeline only'
    )

    # Model configuration
    parser.add_argument(
        '--ml-config',
        type=str,
        metavar='PATH',
        help='Path to model configuration YAML file (e.g., config/models/gaia_2mass_ir.yaml)'
    )

    # Prediction configuration
    parser.add_argument(
        '--pred-config',
        type=str,
        metavar='PATH',
        help='Path to prediction configuration YAML file (e.g., config/prediction/predict_gaia_2mass_ir.yaml)'
    )

    # Validation configuration
    parser.add_argument(
        '--val-config',
        type=str,
        metavar='PATH',
        help='Path to validation configuration YAML file (e.g., config/validation/validate_gaia_2mass_ir.yaml)'
    )

    # Model parameters (legacy, for backward compatibility)
    parser.add_argument(
        '--n-estimators',
        type=int,
        help='Number of trees in Random Forest (default: from config) - LEGACY, use --ml-config instead'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        help='Maximum depth of trees (default: from config) - LEGACY, use --ml-config instead'
    )

    # Options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show configuration
    config = get_config()
    logger.info("Configuration loaded:")
    logger.info(f"  Project root: {config.project_root}")
    logger.info("")

    # Run selected pipeline
    try:
        if args.all:
            run_complete_pipeline(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                model_config=args.ml_config,
                dry_run=args.dry_run
            )

        elif args.data:
            run_data_pipeline(dry_run=args.dry_run)

        elif args.ml:
            run_ml_pipeline(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                model_config=args.ml_config,
                dry_run=args.dry_run
            )

        elif args.predict:
            run_prediction_pipeline(
                pred_config=args.pred_config,
                dry_run=args.dry_run
            )

        elif args.validate:
            run_validation_pipeline(
                val_config=args.val_config,
                dry_run=args.dry_run
            )

        if not args.dry_run:
            logger.info("")
            logger.info("✓ Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
