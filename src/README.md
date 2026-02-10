# Source package (`src`)

Python packages and modules used by the pipeline, scripts, and notebooks. Run from project root so that `src` is on the path (or set `PYTHONPATH`).

| Package / module | Purpose |
|------------------|--------|
| **config** | Central configuration: `get_config()`, paths, dataset keys. See `docs/CONFIGURATION.md`. |
| **features** | Feature engineering for ML (colors, polynomials, etc.). Used by training pipeline and notebooks. |
| **huggingface** | Upload/download datasets and models to/from HuggingFace Hub; README upload. Used by `scripts/download_datasets.py`, `scripts/upload_to_huggingface.py`, `scripts/upload_readmes.py`. See `docs/DATASET_ACCESS.md`. |
| **pipeline** | Pipeline implementations: base classes, configurable ML training, data/ML/prediction/validation. Used by `pipeline.py`. See `docs/PIPELINES.md`, `docs/CONFIGURABLE_PIPELINE.md`. |
| **visualization** | Plotting utilities (e.g. validation plots). |

Pipeline modules under `src/pipeline/`:

- `base.py` — Base classes for pipeline steps and orchestration
- `configurable_ml_pipeline.py` — YAML-driven ML training (Optuna, cache)
- `data_pipeline.py` — Data processing steps
- `ml_pipeline.py` — Legacy ML pipeline
- `prediction_pipeline.py` — Run predictions from prediction configs
- `validation_pipeline.py` — Run validation from validation configs

All paths and dataset names should come from `config/config.yaml` via `get_config()`; see `config/README.md` and `docs/CONFIGURATION.md`.
