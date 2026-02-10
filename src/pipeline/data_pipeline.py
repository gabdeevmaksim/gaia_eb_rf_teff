"""
Data processing pipeline for eclipsing binary temperature analysis.

This pipeline orchestrates all data preparation steps:
1. Convert ECSV to Parquet
2. Extract Pan-STARRS duplicates
3. Merge duplicates
4. Clean photometry
5. Calculate temperatures
6. Add colors and temperatures
7. Prepare ML dataset
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Import our script functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep


class ConvertECSVStep(PipelineStep):
    """Convert ECSV catalog to Parquet format."""

    def __init__(self):
        super().__init__("Convert ECSV to Parquet")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from scripts.convert_ecsv_to_parquet import convert_ecsv_to_parquet

        config = context['config']
        input_file = config.get_dataset_path('eb_catalog', 'raw')
        output_file = config.get_dataset_path('eb_catalog_parquet', 'processed')

        if not input_file.exists():
            self.logger.warning(f"Input file not found: {input_file}")
            self.logger.info("Skipping step...")
            context['eb_catalog_converted'] = False
            return context

        convert_ecsv_to_parquet(input_file, output_file, verbose=False)
        context['eb_catalog_file'] = output_file
        context['eb_catalog_converted'] = True

        return context


class ExtractDuplicatesStep(PipelineStep):
    """Extract Pan-STARRS duplicate entries."""

    def __init__(self):
        super().__init__("Extract Pan-STARRS Duplicates")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from scripts.extract_panstarrs_duplicates import extract_duplicates

        config = context['config']
        input_file = config.get_dataset_path('panstarrs_phot', 'external')
        output_file = config.get_dataset_path('panstarrs_duplicates', 'external')

        if not input_file.exists():
            self.logger.warning(f"Input file not found: {input_file}")
            self.logger.info("Skipping step...")
            context['duplicates_extracted'] = False
            return context

        extract_duplicates(input_file, output_file)
        context['duplicates_file'] = output_file
        context['duplicates_extracted'] = True

        return context


class MergeDuplicatesStep(PipelineStep):
    """Merge Pan-STARRS duplicate entries."""

    def __init__(self):
        super().__init__("Merge Pan-STARRS Duplicates")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from scripts.merge_panstarrs_duplicates_fast import merge_duplicates_fast

        config = context['config']
        input_file = config.get_dataset_path('panstarrs_duplicates', 'external')
        output_file = config.get_dataset_path('panstarrs_duplicates_merged', 'external')
        missing_value = config.get('processing', 'missing_value')

        if not input_file.exists():
            self.logger.warning(f"Input file not found: {input_file}")
            self.logger.info("Skipping step...")
            context['duplicates_merged'] = False
            return context

        merge_duplicates_fast(input_file, output_file, missing_value)
        context['merged_file'] = output_file
        context['duplicates_merged'] = True

        return context


class CleanPhotometryStep(PipelineStep):
    """Clean Pan-STARRS photometry data."""

    def __init__(self):
        super().__init__("Clean Pan-STARRS Photometry")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Import here to avoid circular dependencies
        import pandas as pd
        import numpy as np
        import polars as pl

        config = context['config']
        data_dir = config.get_path('external')
        original_file = config.get_dataset_path('panstarrs_phot', 'external')
        merged_file = config.get_dataset_path('panstarrs_duplicates_merged', 'external')
        output_file = config.get_dataset_path('panstarrs_cleaned', 'external')
        missing_val = config.get('processing', 'missing_value')

        # Load data
        df_original = pl.read_csv(original_file)
        df_merged = pl.read_csv(merged_file)

        # Remove duplicates from original
        merged_source_ids = set(df_merged['original_ext_source_id'])
        df_no_duplicates = df_original.filter(
            ~pl.col('original_ext_source_id').is_in(merged_source_ids)
        )

        # Combine
        df_combined = pl.concat([df_no_duplicates, df_merged])

        # Filter for magnitude pairs
        has_g_r = (df_combined['gPSFMag'] != missing_val) & (df_combined['rPSFMag'] != missing_val)
        has_r_i = (df_combined['rPSFMag'] != missing_val) & (df_combined['iPSFMag'] != missing_val)
        has_i_z = (df_combined['iPSFMag'] != missing_val) & (df_combined['zPSFMag'] != missing_val)

        mask = has_g_r | has_r_i | has_i_z
        df_filtered = df_combined.filter(mask)

        # Calculate colors
        df_filtered = df_filtered.with_columns([
            pl.when((pl.col('gPSFMag') != missing_val) & (pl.col('rPSFMag') != missing_val))
              .then(pl.col('gPSFMag') - pl.col('rPSFMag'))
              .otherwise(missing_val)
              .alias('g_r_color'),

            pl.when((pl.col('rPSFMag') != missing_val) & (pl.col('iPSFMag') != missing_val))
              .then(pl.col('rPSFMag') - pl.col('iPSFMag'))
              .otherwise(missing_val)
              .alias('r_i_color'),

            pl.when((pl.col('iPSFMag') != missing_val) & (pl.col('zPSFMag') != missing_val))
              .then(pl.col('iPSFMag') - pl.col('zPSFMag'))
              .otherwise(missing_val)
              .alias('i_z_color')
        ])

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.write_csv(output_file)

        self.logger.info(f"Cleaned photometry: {len(df_filtered):,} objects")
        context['cleaned_file'] = output_file
        context['photometry_cleaned'] = True

        return context


class CalculateTemperaturesStep(PipelineStep):
    """Calculate effective temperatures from colors."""

    def __init__(self):
        super().__init__("Calculate Temperatures")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from scripts.calculate_temperatures import main as calculate_temps

        # Run the script (it uses config internally)
        calculate_temps()

        config = context['config']
        output_file = config.get_dataset_path('panstarrs_with_temps', 'external')

        context['temperatures_file'] = output_file
        context['temperatures_calculated'] = True

        return context


class DataProcessingPipeline(Pipeline):
    """
    Complete data processing pipeline.

    Steps:
    1. Convert ECSV to Parquet
    2. Extract Pan-STARRS duplicates
    3. Merge duplicates
    4. Clean photometry
    5. Calculate temperatures

    Usage
    -----
    >>> pipeline = DataProcessingPipeline()
    >>> context = pipeline.run()
    """

    def __init__(self):
        steps = [
            ConvertECSVStep(),
            ExtractDuplicatesStep(),
            MergeDuplicatesStep(),
            CleanPhotometryStep(),
            CalculateTemperaturesStep(),
        ]

        super().__init__("Data Processing Pipeline", steps)
