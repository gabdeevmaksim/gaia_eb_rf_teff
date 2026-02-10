"""
Pipeline orchestration for eclipsing binary temperature analysis.

This module provides pipeline classes that orchestrate the data processing
and machine learning workflows.
"""

from .base import Pipeline, PipelineStep
from .data_pipeline import DataProcessingPipeline
from .ml_pipeline import MLTrainingPipeline
from .configurable_ml_pipeline import ConfigurableMLPipeline
from .prediction_pipeline import PredictionPipeline
from .validation_pipeline import ValidationPipeline

__all__ = [
    'Pipeline',
    'PipelineStep',
    'DataProcessingPipeline',
    'MLTrainingPipeline',
    'ConfigurableMLPipeline',
    'PredictionPipeline',
    'ValidationPipeline'
]
