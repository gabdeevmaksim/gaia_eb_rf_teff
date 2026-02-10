# Documentation

Index of project documentation. Paths and config names reflect the current repo: input `data/raw/eb_unified_photometry`, output `data/processed/eb_catalog_teff`, config keys in `config/config.yaml`.

## Pipeline and configuration

| File | Purpose |
|------|--------|
| **PIPELINES.md** | Pipeline architecture, `pipeline.py` usage, data/ML/prediction/validation steps. |
| **CONFIGURABLE_PIPELINE.md** | Configurable ML pipeline, YAML-driven training and prediction. |
| **CONFIGURATION.md** | Central config (`config/config.yaml`), `get_config()`, `get_dataset_path()`. |
| **CONFIG_TEMPLATE_GUIDE.md** | Template for `config/models/template_training_config.yaml`. |
| **OPTUNA_HYPERPARAMETER_OPTIMIZATION.md** | Optuna usage, cache, search space. |

## Data, deployment, and HuggingFace

| File | Purpose |
|------|--------|
| **DATASET_ACCESS.md** | Download and use of datasets and models from HuggingFace. |
| **DEPLOYMENT_SUMMARY.md** | Deployment snapshot and setup. |
| **DOCKER_USAGE.md** | Docker build/run, compose, troubleshooting. |
| **README_DEPLOY.md** | Deployment-focused quick start (Docker, local, config). |
| **HF_DATASET_README.md** | Content uploaded to the HuggingFace dataset repo as README. |
| **HF_MODEL_README.md** | Content uploaded to the HuggingFace model repo as README. |

## Config READMEs (reference)

- `config/README.md` — Root config files (`config.yaml`, `hyperparameter_cache.yaml`).
- `config/models/README.md` — Training configs and model registry.
- `config/prediction/README.md` — Prediction configs.
- `config/validation/README.md` — Validation configs.
