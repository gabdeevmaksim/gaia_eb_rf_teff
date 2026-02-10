# Prediction configuration

YAML configs for running trained models on data. Each config specifies which model, which data, and how to write outputs.

## Usage

```bash
python pipeline.py predict --config config/prediction/predict_gaia_teff_corrected_log_optuna.yaml
```

Or from Python:

```python
from src.pipeline import PredictionPipeline
pipeline = PredictionPipeline('config/prediction/predict_gaia_teff_corrected_log_optuna.yaml')
context = pipeline.run()
```

## Available configs

| Config | Purpose |
|--------|--------|
| `predict_gaia_teff_corrected_log_optuna.yaml` | Teff from Gaia colors (log model); objects **without** Gaia Teff. |
| `predict_gaia_teff_corrected_clustering_optuna.yaml` | Teff with clustering features; objects without Gaia Teff. |
| `predict_gaia_teff_flag1_corrected_optuna.yaml` | Teff flag1 model; no Teff filter. |
| `predict_gaia_original_teff_flag1_corrected_optuna.yaml` | Teff flag1; only objects **with** Gaia Teff (validation). |
| `predict_gaia_logg.yaml` | Surface gravity (logg) from Gaia colors. |
| `predict_teff_logg_only_no_teff.yaml` | Teff from Gaia + logg (chain); objects without Teff. Requires `eb_unified_photometry_with_logg.parquet` (run `enrich_dataset_with_logg_predictions.py` first). |
| `template_prediction.yaml` | Template to copy for new prediction configs. |

Paths (e.g. `data.source_file`) are resolved from project root; data dir comes from `config/config.yaml` when defined there.

## Main schema

```yaml
model:
  model_file: "rf_model_*.pkl"   # Wildcard = latest by timestamp; or exact filename

data:
  source_file: "eb_unified_photometry.parquet"
  source_location: raw          # Input from data/raw (from HF); use "processed" for data/processed
  id_column: "source_id"
  filters: {}                   # Optional: teff_gaia_lt: 0.1, teff_gaia_gt: 0.1, teff_gaia: -999.0, etc.

preprocessing:
  filter_missing: true
  missing_value: -999.0
  required_features: ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"]   # Must match model

feature_engineering:
  enabled: false                # Must match training config if model used it

target_transform:               # If model predicted log10(Teff)
  type: "log10"
  inverse: true

uncertainty:
  enabled: true
  method: "full_tree"           # or "fast"
  n_sample_trees: null          # null = all trees

output:
  output_file: "predictions.parquet"
  include_columns: ["source_id"]
  save_summary: true
```

Optional: `clustering` (e.g. for clustering-based Teff models), with `enabled`, `method`, `features`.

## Adding a new prediction config

1. Copy the template:  
   `cp config/prediction/template_prediction.yaml config/prediction/my_predict.yaml`
2. Set `model.model_file` to the `.pkl` in `models/` (wildcard or exact).
3. Set `data.source_file`, `data.id_column`; add `data.filters` if needed.
4. Set `preprocessing.required_features` to the **exact** list the model was trained on.
5. Set `feature_engineering` to match the training config (same `enabled`, `color_cols`/`mag_cols` if used).
6. If the model predicts log10(Teff), add `target_transform: { type: "log10", inverse: true }`.
7. Set `output.output_file` and `output.include_columns`.

## Important

- **Feature engineering** must match training (same `enabled`, same columns). Mismatch gives wrong predictions.
- **Model wildcard** (e.g. `rf_gaia_teff_corrected_log_optuna_*.pkl`) picks the **most recent** matching file in `models/` by filename. Use an exact filename for a fixed version.
- **Output format** is inferred from `output_file` extension (e.g. `.parquet`, `.csv`).

## Outputs

- **Predictions file** (parquet/CSV): prediction column(s), e.g. `teff_predicted` or `logg_predicted`, plus `include_columns`. Uncertainty columns if `uncertainty.enabled: true`.
- **Summary file** (`.txt`): model path, input file, object count, prediction stats; written when `output.save_summary: true`.

## Troubleshooting

| Issue | Check |
|-------|--------|
| Model not found | `ls models/rf_*.pkl`; fix `model_file` or train first. |
| Data not found | Path relative to project root; data dir from `config/config.yaml` if set. |
| Missing required features | `required_features` must list every column the model needs; add columns to data or use another model. |
| Unrealistic values | Feature engineering and target_transform must match the training config. |

## See also

- `config/models/README.md` – Training configs and model registry.
- `src/pipeline/prediction_pipeline.py` – Prediction pipeline implementation.
