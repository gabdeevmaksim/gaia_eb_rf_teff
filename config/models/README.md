# Model training configuration

This directory holds **training configs** (YAML) and the **HuggingFace model registry**.

## Contents

| File | Purpose |
|------|--------|
| `template_training_config.yaml` | Full template with every option documented. Copy and edit to add a new model. |
| `model_registry.yaml` | List of models published on HuggingFace; used by upload/download scripts. |
| `gaia_*.yaml` | Concrete training configs (Teff, logg, clustering, flag1, etc.). |

## Training a model

```bash
python pipeline.py --ml-config config/models/gaia_teff_corrected_log_optuna.yaml
```

Paths in configs are relative to the **project root**. Data paths (e.g. `data.source_file`) are resolved via `config/config.yaml` when a key exists there; otherwise they are relative to project root.

## Adding a new model

1. Copy `template_training_config.yaml` to e.g. `config/models/my_model.yaml`.
2. Set at least: `model.name`, `model.id_prefix`, `data.source_file`, `data.target`, `data.features`, `preprocessing.missing_value`.
3. Run: `python pipeline.py --ml-config config/models/my_model.yaml`.
4. To publish: add an entry to `model_registry.yaml` and use `scripts/upload_to_huggingface.py --models <key>`.

## Main config sections

- **model** – Name, id_prefix, description.
- **teff_correction** – Optional polynomial correction for Teff > threshold (e.g. 10 000 K).
- **target_transform** – `none`, `log`, `log2`, `ln` (use `log` for Teff).
- **data** – Source file, target column, feature list, id_column.
- **preprocessing** – `missing_value`, `filters`, `drop_missing`.
- **feature_engineering** – Optional; must match prediction config if used.
- **training** – `test_size`, `random_state`.
- **optuna_optimization** – Optional; hyperparameters cached by config signature.
- **hyperparameters** – RF params (used as defaults or overwritten by Optuna).
- **clustering** – Optional cluster-feature pipeline (e.g. GMM/KMeans).
- **output** – Where to save model and predictions.

## Model registry (HuggingFace)

`model_registry.yaml` lists models that are (or will be) on the Hub. Each entry has:

- `file` – Filename in `models/` (e.g. `rf_gaia_teff_corrected_log_20251126_130144.pkl`).
- `description`, `features`, `target`, `performance`, `url`.

Upload: `python scripts/upload_to_huggingface.py --models <registry_key>` or `--models all`.  
Download: `python scripts/download_datasets.py --model <registry_key>` or `--model all`.

## See also

- **Prediction configs** – `config/prediction/` (which model file and data to use for inference).
- **Validation configs** – `config/validation/` (evaluation pipelines).
- **Pipeline** – `src/pipeline/configurable_ml_pipeline.py` (how training configs are loaded and run).
