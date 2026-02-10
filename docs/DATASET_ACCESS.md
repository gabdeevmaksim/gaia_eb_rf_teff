# Dataset Access Guide

Complete guide for downloading and using the Gaia EB Teff datasets and pre-trained models.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Datasets Available](#datasets-available)
- [Download Methods](#download-methods)
- [Pre-trained Models](#pre-trained-models)
- [Authentication](#authentication)
- [Dataset Structure](#dataset-structure)
- [Usage Examples](#usage-examples)

## Overview

All datasets and models are hosted on HuggingFace Hub for easy access, version control, and distribution.

**Repositories**:
- **Datasets**: [Dedulek/gaia-eb-teff-datasets](https://huggingface.co/datasets/Dedulek/gaia-eb-teff-datasets)
- **Models**: [Dedulek/gaia-eb-teff-models](https://huggingface.co/models/Dedulek/gaia-eb-teff-models)

## Quick Start

```bash
# Install HuggingFace Hub
pip install huggingface_hub

# Set token (optional for public datasets)
export HF_TOKEN=your_token_here

# Download input photometry (saves to data/raw/)
python scripts/download_datasets.py --datasets training

# Download a model (saves to models/)
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

## Datasets Available

### 1. Input: Unified photometry (pipeline input)

#### photometry/eb_unified_photometry.parquet

**Description**: Unified multi-survey photometry for ~2.18M eclipsing binary stars (Gaia + Pan-STARRS + 2MASS).

**Download**: `python scripts/download_datasets.py --datasets training` → files land in **`data/raw/`**.

**Use**: Input for training and prediction. Training configs use `source_location: raw` and `source_file: eb_unified_photometry.parquet`.

### 2. Output: Catalog with Teff (pipeline output)

#### predictions/eb_catalog_teff.parquet

**Description**: Eclipsing binary catalog with Teff: Gaia GSP-Phot (original + corrected for Teff > 10k K) plus ML predictions from four models (clustering, log, flag1, chain), with `teff_best` and `teff_best_uncertainty` per object.

**Produced by**: `scripts/merge_teff_predictions_into_unified.py` (run after prediction parquets exist in `data/processed/`).

**Columns** (summary): `source_id`, `teff_gaia_original`, `teff_gaia_corrected`, model columns (`teff_clustering`, `teff_log`, `teff_flag1`, `teff_propagated` and their uncertainties), `teff_best`, `teff_best_uncertainty`.

### 3. Other datasets on the Hub

- **catalog**: `catalogs/stars_types_with_best_predictions.*` (if present on Hub).
- **correction**: `correction/teff_correction_coeffs_deg2.pkl` — download with `--datasets correction` (saves to `data/`).

## Download Methods

### Method 1: Using Provided Script (Recommended)

```bash
# Download input photometry (saves to data/raw/)
python scripts/download_datasets.py --datasets training

# Download correction coefficients (saves to data/)
python scripts/download_datasets.py --datasets correction

# Download catalog (if present on Hub)
python scripts/download_datasets.py --datasets catalog

# Download specific model (saves to models/)
python scripts/download_datasets.py --model gaia_teff_corrected_log

# Download all models
python scripts/download_datasets.py --model all
```

**Advantages**: Automatic paths (`training` → `data/raw/`, `correction` → `data/`), error handling.

### Method 2: HuggingFace Hub Python API

```python
from huggingface_hub import hf_hub_download

# Download input photometry (for pipeline input)
phot_path = hf_hub_download(
    repo_id="Dedulek/gaia-eb-teff-datasets",
    filename="photometry/eb_unified_photometry.parquet",
    repo_type="dataset"
)
# Save to data/raw/eb_unified_photometry.parquet for pipeline use

import polars as pl
df = pl.read_parquet(phot_path)
```

### Method 3: HuggingFace CLI

```bash
# Install CLI
pip install huggingface_hub[cli]

# Login (optional for public datasets)
huggingface-cli login

# Download specific file
huggingface-cli download YOUR_ORG/gaia-eb-teff-datasets \
    catalogs/stars_types_with_best_predictions.fits \
    --repo-type dataset \
    --local-dir data/processed

# Download all training data
huggingface-cli download YOUR_ORG/gaia-eb-teff-datasets \
    --include "training/*" \
    --repo-type dataset \
    --local-dir data/processed
```

### Method 4: Git LFS (For Complete Repository)

```bash
# Clone entire dataset repository
git clone https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets

# Or sparse checkout (specific files only)
git clone --no-checkout https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets
cd gaia-eb-teff-datasets
git sparse-checkout set catalogs/
git checkout
```

**Note**: Requires Git LFS for large files. Install with: `git lfs install`

### Method 5: Direct URL Download

```bash
# Catalog
wget https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/catalogs/stars_types_with_best_predictions.fits

# Training data
wget https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/training/gaia_all_colors_teff_corrected.parquet

# Using curl
curl -L -o catalog.fits https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/catalogs/stars_types_with_best_predictions.fits
```

## Pre-trained Models

All models are available in the [model repository](https://huggingface.co/models/YOUR_ORG/gaia-eb-teff-models).

### Available Models

#### 1. gaia_teff_corrected_log (RECOMMENDED)

**File**: `rf_gaia_teff_corrected_log_20251126_130144.pkl` (1.2 GB)

**Description**: Best overall model - log-transformed Gaia colors with corrected Teff

**Performance**:
- MAE: 556.9K
- RMSE: 1021.3K
- R²: 0.640
- Within 10%: 68.5%

**Features**: 6 Gaia colors + 3 bands (BP, RP, G)

**Training**: 1.27M stars, corrected for Teff > 10,000K

**Download**:
```bash
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

#### 2. gaia_2mass_ir

**File**: `rf_gaia_2mass_ir_20251103_141119.pkl` (1.2 GB)

**Description**: Gaia optical + 2MASS infrared photometry

**Performance**:
- MAE: 765.1K
- RMSE: 1168.4K
- R²: 0.315
- Within 10%: 43.4%

**Features**: Gaia colors + 2MASS J, H, K bands

**Use Case**: When infrared data is available

#### 3. gaia_all_colors_teff_log

**File**: `rf_gaia_all_colors_teff_log_20251112_162857.pkl` (2.0 GB)

**Description**: All Gaia colors, log-transformed target

**Performance**: Similar to gaia_teff_corrected_log

**Difference**: Uses uncorrected Gaia Teff (for comparison)

### Model Registry

All models are cataloged in `config/models/model_registry.yaml`:

```yaml
models:
  gaia_teff_corrected_log:
    file: "rf_gaia_teff_corrected_log_20251126_130144.pkl"
    url: "https://huggingface.co/.../rf_gaia_teff_corrected_log_20251126_130144.pkl"
    checksum: "sha256:..."
    mae_kelvin: 556.9
    features: [bp_rp, g_rp, g_bp, bp_g, rp_g, bp, rp, g]
```

### Model Files

Each model includes:
- **`.pkl`**: Trained model (scikit-learn RandomForest)
- **`_metadata.json`**: Features, hyperparameters, training info
- **`_SUMMARY.txt`**: Human-readable performance summary

## Authentication

### Public Datasets

Most datasets are **public** and don't require authentication. You can download without a token.

### Private Datasets / Models

If datasets are private, you'll need a HuggingFace token:

#### Option 1: CLI Login

```bash
huggingface-cli login
# Enter your token when prompted
```

#### Option 2: Environment Variable

```bash
export HF_TOKEN=your_huggingface_token_here
```

#### Option 3: Python Code

```python
from huggingface_hub import login
login(token="your_token_here")
```

#### Getting a Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access sufficient)
3. Copy and save securely

## Dataset Structure

**Datasets repo** (`Dedulek/gaia-eb-teff-datasets`):
- `photometry/eb_unified_photometry.parquet` — input photometry (download with `--datasets training` → `data/raw/`)
- `photometry/eb_unified_photometry_SUMMARY.txt`
- `predictions/eb_catalog_teff.parquet` — catalog with Teff (produced by merge script)
- `predictions/eb_catalog_teff_SUMMARY.txt`
- `correction/teff_correction_coeffs_deg2.pkl` — download with `--datasets correction` → `data/`
- `catalogs/` — optional (if present)

**Models repo** (`Dedulek/gaia-eb-teff-models`):
- `model_registry.yaml` — model manifest
- `rf_*.pkl`, `*_metadata.json`, `*_SUMMARY.txt` — per model

## Usage Examples

### Example 1: Load input photometry or catalog in Python

```python
import polars as pl

# After download: data/raw/eb_unified_photometry.parquet
df = pl.read_parquet('data/raw/eb_unified_photometry.parquet')

# Or catalog with Teff (after merge): data/processed/eb_catalog_teff.parquet
catalog = pl.read_parquet('data/processed/eb_catalog_teff.parquet')
teff_best = catalog['teff_best']

# Filter by uncertainty (column: teff_best_uncertainty)
low_unc = catalog.filter(pl.col('teff_best_uncertainty') < 300)

# Statistics
print(f"Total stars: {len(catalog)}")
print(f"Mean Teff: {catalog['teff_best'].mean():.0f} K")
print(f"Mean uncertainty: {catalog['teff_best_uncertainty'].mean():.0f} K")
```

### Example 2: Train via pipeline (recommended)

Use the project pipeline and config so paths and features stay consistent:

```bash
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log_optuna.yaml
```

Input data: `data/raw/eb_unified_photometry.parquet` (download with `scripts/download_datasets.py --datasets training`).

### Example 3: Make predictions via pipeline

```bash
python pipeline.py --predict --pred-config config/prediction/predict_gaia_teff_flag1_corrected_optuna.yaml
```

Or in Python: load model from `models/` (see `config/models/`) and use `required_features` from the matching prediction config in `config/prediction/`.

### Example 4: Docker Auto-Download

```bash
# Datasets download automatically in Docker containers
export HF_TOKEN=your_token
docker compose run --rm train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml
# Container downloads training data if not present
```

## File Formats

### Parquet Files

**Advantages**: Fast I/O, columnar storage, efficient compression

**Reading**:
```python
# Polars (recommended)
import polars as pl
data = pl.read_parquet('file.parquet')

# Pandas
import pandas as pd
data = pd.read_parquet('file.parquet')

# PyArrow
import pyarrow.parquet as pq
table = pq.read_table('file.parquet')
```

### FITS Files

**Advantages**: Standard astronomy format, header metadata

**Reading**:
```python
# Astropy (recommended)
from astropy.table import Table
catalog = Table.read('file.fits')

# Astropy.io.fits
from astropy.io import fits
hdul = fits.open('file.fits')
data = hdul[1].data  # Binary table
```

## License

All datasets and models are released under **CC BY 4.0** license.

You are free to:
- Share and redistribute
- Adapt and build upon
- Use for commercial purposes

With attribution required.

## Citation

If you use these datasets or models, please cite:

```bibtex
@article{your_paper,
  title={Effective Temperature Predictions for Eclipsing Binary Stars},
  author={Your Name},
  journal={Journal},
  year={2025},
  note={Dataset: https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets}
}
```

## Support

For issues with datasets or models:
- **GitHub Issues**: https://github.com/YOUR_ORG/gaia-eb-teff/issues
- **HuggingFace Discussions**: Use the discussion tab on the dataset page
- **Email**: your.email@example.com

## Updates

Dataset versions are tracked on HuggingFace Hub. Check the repository for updates:
- https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets
- https://huggingface.co/models/YOUR_ORG/gaia-eb-teff-models
