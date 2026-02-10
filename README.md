# Gaia eclipsing binary Teff pipeline

ML-based effective temperature prediction for eclipsing binaries using Gaia (and optional) photometry. Trains Random Forest models from config-driven YAMLs, runs prediction and validation, and produces a merged catalog with Teff estimates.

## Quick start

```bash
# Install
pip install -r requirements.txt

# Get input data (photometry → data/raw/)
python scripts/download_datasets.py --datasets training

# Train a model
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log_optuna.yaml

# Predict (writes to data/processed/)
python pipeline.py --predict --pred-config config/prediction/predict_gaia_teff_flag1_corrected_optuna.yaml

# Validate
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_flag1_corrected_optuna.yaml
```

Merge all prediction outputs into the final catalog:

```bash
python scripts/merge_teff_predictions_into_unified.py
# → data/processed/eb_catalog_teff.parquet
```

## Root layout

| Path | Purpose |
|------|--------|
| **pipeline.py** | CLI: `--ml`, `--predict`, `--validate`, `--data`, `--all`; use `--ml-config`, `--pred-config`, `--val-config` for YAML configs. |
| **config/** | Central `config.yaml`, `hyperparameter_cache.yaml`, and subdirs: `models/`, `prediction/`, `validation/`. See [config/README.md](config/README.md). |
| **src/** | Python packages: config, features, huggingface, pipeline, visualization. See [src/README.md](src/README.md). |
| **scripts/** | One-off scripts: download/upload HuggingFace, merge predictions, evaluation, paper figures. |
| **data/raw/** | Input photometry (e.g. `eb_unified_photometry.parquet` from download). |
| **data/processed/** | Prediction parquets and merged catalog `eb_catalog_teff.parquet`. |
| **models/** | Trained `.pkl` models and metadata. |
| **docs/** | Full documentation index in [docs/README.md](docs/README.md). |
| **paper/** | Manuscript figures and model comparison tables. |

## Configuration and docs

- **Paths and datasets:** [config/README.md](config/README.md) and [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
- **Pipelines and configurable ML:** [docs/PIPELINES.md](docs/PIPELINES.md), [docs/CONFIGURABLE_PIPELINE.md](docs/CONFIGURABLE_PIPELINE.md).
- **Data and HuggingFace:** [docs/DATASET_ACCESS.md](docs/DATASET_ACCESS.md).
- **Docker:** [docs/DOCKER_USAGE.md](docs/DOCKER_USAGE.md).

All documentation lives under **docs/**; see [docs/README.md](docs/README.md) for the full list.
