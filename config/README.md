# Configuration

Central configuration for paths, datasets, processing, and ML. Subdirectories hold task-specific YAML configs.

## Root-level files

| File | Purpose |
|------|--------|
| **config.yaml** | Main project config: `paths` (data/raw, data/processed, models, reports), `datasets` (input in raw: `eb_unified_photometry`; output in processed: `eb_catalog_teff`), `processing`, `ml`, `temperature`, `logging`. Used by `src.config.get_config()` and pipelines. |
| **hyperparameter_cache.yaml** | Optuna cache: stores best hyperparameters per model signature so re-runs skip optimization. Written by the ML pipeline; don’t edit by hand. See `config/models/README.md` (Optuna section). |

## Subdirectories

| Directory | Contents |
|-----------|----------|
| **models/** | Training configs and model registry. See `config/models/README.md`. |
| **prediction/** | Prediction configs (which model, which data, output). See `config/prediction/README.md`. |
| **validation/** | Validation configs (metrics and plots). See `config/validation/README.md`. |
