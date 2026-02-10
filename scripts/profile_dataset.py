#!/usr/bin/env python3
"""
Generate data profiling reports using ydata_profiling.

This script creates comprehensive HTML reports for datasets, including:
- Overview statistics
- Variable types and distributions
- Missing values analysis
- Correlations
- Sample data

Config dataset keys (see config/config.yaml): eb_unified_photometry, eb_catalog_teff.
Use --location raw for input photometry, processed for the merged catalog.

Usage:
    # Profile input photometry (from config, raw location)
    python scripts/profile_dataset.py --dataset eb_unified_photometry --location raw

    # Profile merged catalog with Teff (from config, processed)
    python scripts/profile_dataset.py --dataset eb_catalog_teff --location processed

    # Profile a specific file
    python scripts/profile_dataset.py --file data/raw/eb_unified_photometry.parquet

    # Profile all datasets from config (tries raw/processed/external/interim per key)
    python scripts/profile_dataset.py --all

    # Custom output directory
    python scripts/profile_dataset.py --dataset eb_catalog_teff --output reports/profiles
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ydata_profiling import ProfileReport
except ImportError:
    print("Error: ydata_profiling not installed. Install with: pip install ydata_profiling")
    sys.exit(1)

import pandas as pd
import polars as pl
from src.config import get_config


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from file path, supporting multiple formats.
    
    Parameters
    ----------
    file_path : Path
        Path to dataset file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset as pandas DataFrame
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.parquet':
        # Try polars first (faster), fallback to pandas
        try:
            df_pl = pl.read_parquet(file_path)
            return df_pl.to_pandas()
        except:
            return pd.read_parquet(file_path)
    elif suffix == '.csv':
        return pd.read_csv(file_path)
    elif suffix == '.fits':
        from astropy.table import Table
        table = Table.read(str(file_path))
        return table.to_pandas()
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .parquet, .csv, .fits")


def handle_missing_values(df: pd.DataFrame, missing_value: float = -999.0) -> pd.DataFrame:
    """
    Replace missing value indicators with NaN for proper profiling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    missing_value : float, default=-999.0
        Value that represents missing data
        
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values replaced by NaN
    """
    df_clean = df.copy()
    
    # Count missing values before replacement
    missing_count = (df_clean == missing_value).sum().sum()
    
    if missing_count > 0:
        print(f"  Replacing {missing_count:,} missing value indicators ({missing_value}) with NaN...")
        # Replace missing value indicator with NaN
        df_clean = df_clean.replace(missing_value, pd.NA)
        
        # For numeric columns, also convert to float to properly handle NaN
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    else:
        print(f"  No missing value indicators ({missing_value}) found.")
    
    return df_clean


def create_profile(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Dataset Profiling Report",
    minimal: bool = False,
    missing_value: float = -999.0
) -> Path:
    """
    Create a profiling report for a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to profile
    output_path : Path
        Output file path for HTML report
    title : str
        Title for the report
    minimal : bool
        If True, create minimal report (faster for large datasets)
    missing_value : float, default=-999.0
        Value that represents missing data (will be replaced with NaN)
        
    Returns
    -------
    Path
        Path to generated report
    """
    print(f"  Creating profile report for {len(df):,} rows × {len(df.columns)} columns...")
    
    # Handle missing values before profiling
    df_clean = handle_missing_values(df, missing_value=missing_value)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate profile report
    profile = ProfileReport(
        df_clean,
        title=title,
        minimal=minimal,
        progress_bar=True
    )
    
    # Save report
    profile.to_file(output_path)
    
    print(f"  ✓ Report saved to: {output_path}")
    
    return output_path


def profile_from_config(dataset_key: str, output_dir: Path, location: str = 'processed', missing_value: float = None):
    """
    Profile a dataset using config key.
    
    Parameters
    ----------
    dataset_key : str
        Dataset key from config.yaml
    output_dir : Path
        Output directory for reports
    location : str
        Data location (raw, processed, external, interim)
    missing_value : float, optional
        Missing value indicator (default: from config)
    """
    config = get_config()
    
    # Get missing value from config if not provided
    if missing_value is None:
        missing_value = config.get('processing', 'missing_value', -999.0)
    
    try:
        file_path = config.get_dataset_path(dataset_key, location)
    except KeyError:
        print(f"Error: Dataset key '{dataset_key}' not found in config.yaml")
        print(f"Available datasets: {list(config._config.get('datasets', {}).keys())}")
        return False
    
    print(f"\nProfiling dataset: {dataset_key}")
    print(f"  File: {file_path}")
    print(f"  Missing value indicator: {missing_value}")
    
    # Load dataset
    try:
        df = load_dataset(file_path)
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False
    
    # Create output filename
    output_filename = f"{dataset_key}_profile.html"
    output_path = output_dir / output_filename
    
    # Generate profile
    try:
        create_profile(df, output_path, title=f"{dataset_key} - Data Profiling Report", missing_value=missing_value)
        return True
    except Exception as e:
        print(f"  ✗ Error creating profile: {e}")
        return False


def profile_from_file(file_path: Path, output_dir: Path, missing_value: float = None):
    """
    Profile a dataset from file path.
    
    Parameters
    ----------
    file_path : Path
        Path to dataset file
    output_dir : Path
        Output directory for reports
    missing_value : float, optional
        Missing value indicator (default: from config)
    """
    config = get_config()
    
    # Get missing value from config if not provided
    if missing_value is None:
        missing_value = config.get('processing', 'missing_value', -999.0)
    
    print(f"\nProfiling file: {file_path}")
    print(f"  Missing value indicator: {missing_value}")
    
    # Load dataset
    try:
        df = load_dataset(file_path)
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False
    
    # Create output filename
    output_filename = f"{file_path.stem}_profile.html"
    output_path = output_dir / output_filename
    
    # Generate profile
    try:
        title = f"{file_path.stem} - Data Profiling Report"
        create_profile(df, output_path, title=title, missing_value=missing_value)
        return True
    except Exception as e:
        print(f"  ✗ Error creating profile: {e}")
        return False


def profile_all_from_config(output_dir: Path, missing_value: float = None):
    """
    Profile all datasets defined in config.yaml.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for reports
    missing_value : float, optional
        Missing value indicator (default: from config)
    """
    config = get_config()
    datasets = config._config.get('datasets', {})
    
    # Get missing value from config if not provided
    if missing_value is None:
        missing_value = config.get('processing', 'missing_value', -999.0)
    
    print(f"\nProfiling {len(datasets)} datasets from config...")
    print(f"Missing value indicator: {missing_value}")
    
    success_count = 0
    for dataset_key in datasets.keys():
        # Try processed first, then other locations
        locations = ['processed', 'raw', 'external', 'interim']
        profiled = False
        
        for location in locations:
            try:
                file_path = config.get_dataset_path(dataset_key, location)
                if file_path.exists():
                    if profile_from_config(dataset_key, output_dir, location, missing_value):
                        success_count += 1
                        profiled = True
                        break
            except (KeyError, FileNotFoundError):
                continue
        
        if not profiled:
            print(f"  ⚠ Skipped {dataset_key} (file not found)")
    
    print(f"\n✓ Successfully profiled {success_count}/{len(datasets)} datasets")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate data profiling reports using ydata_profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Dataset selection (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--dataset',
        help='Dataset key from config.yaml: eb_unified_photometry (use --location raw) or eb_catalog_teff (use --location processed)'
    )
    dataset_group.add_argument(
        '--file',
        type=Path,
        help='Path to dataset file (e.g., data/processed/file.parquet)'
    )
    dataset_group.add_argument(
        '--all',
        action='store_true',
        help='Profile all datasets from config.yaml'
    )
    
    # Options
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/profiles'),
        help='Output directory for reports (default: reports/profiles)'
    )
    parser.add_argument(
        '--location',
        default='processed',
        choices=['raw', 'processed', 'external', 'interim'],
        help='Data location for --dataset option (default: processed)'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Create minimal report (faster for large datasets)'
    )
    parser.add_argument(
        '--missing-value',
        type=float,
        default=None,
        help='Missing value indicator (default: from config.yaml, usually -999.0)'
    )
    
    args = parser.parse_args()
    
    # Get missing value from config if not provided via command line
    if args.missing_value is None:
        config = get_config()
        args.missing_value = config.get('processing', 'missing_value', default=-999.0)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Profile based on selection
    success = False
    
    if args.all:
        success = profile_all_from_config(args.output, args.missing_value)
    elif args.dataset:
        success = profile_from_config(args.dataset, args.output, args.location, args.missing_value)
    elif args.file:
        success = profile_from_file(args.file, args.output, args.missing_value)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
