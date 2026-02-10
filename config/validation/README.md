# Validation configuration

YAML configs for validating trained models: metrics (MAE, RMSE, R², within-% thresholds), performance by temperature bins, and optional plots. Validation uses the model’s **test set predictions** (saved at training time).

## Usage

```bash
python pipeline.py validate --config config/validation/validate_gaia_teff_flag1_corrected_optuna.yaml
```

Or from Python:

```python
from src.pipeline import ValidationPipeline
pipeline = ValidationPipeline('config/validation/validate_gaia_teff_flag1_corrected_optuna.yaml')
context = pipeline.run()
```

## Available configs

| Config | Purpose |
|--------|--------|
| `validate_gaia_teff_flag1_corrected_optuna.yaml` | Validates the flag1 Teff model (latest `rf_gaia_teff_flag1_corrected_optuna_*`). |
| `template_validation.yaml` | Template to copy for validating other models. |

The pipeline expects a matching **test predictions** file in `models/` (e.g. `rf_gaia_teff_flag1_corrected_optuna_*_test_predictions.parquet`). That file is written when you train with the same pipeline.

## Schema

```yaml
model:
  model_pattern: "rf_gaia_teff_flag1_corrected_optuna_*"   # Wildcard = latest; or exact ID

plots:
  test_scatter: true          # Predicted vs actual scatter
  residuals: true              # Residuals vs actual
  performance_by_temp: true    # MAE/RMSE by temperature bins
  temp_distributions: true     # Distribution comparison
  feature_importance: true    # Feature importance (if model has it)
  color_temp_relations: false  # Optional: scatter of a color vs Teff
  color_columns: []           # If color_temp_relations: list of {column, label}

output:
  figures_subdir: "gaia_teff_flag1_validation"   # Under reports/figures/
  report_file: "reports/validation_report_gaia_teff_flag1.txt"

target_info:
  name: "Temperature"
  unit: "K"
  short: "Teff"

# Optional: custom temperature bins for performance_by_temp
# temp_ranges: [[0, 4000], [4000, 5000], [5000, 6000], [6000, 8000], [8000, 50000]]
```

## Adding a new validation config

1. Copy the template:  
   `cp config/validation/template_validation.yaml config/validation/validate_my_model.yaml`
2. Set `model.model_pattern` to the model ID or pattern (e.g. `rf_my_model_*` for latest).
3. Enable/disable plots under `plots`.
4. Set `output.figures_subdir` and `output.report_file`.
5. Adjust `target_info` if the target is not Teff (e.g. logg).
6. Run: `python pipeline.py validate --config config/validation/validate_my_model.yaml`

**Model wildcard:** `rf_my_model_*` selects the most recent matching `.pkl` in `models/` by filename. Use an exact ID (no `*`) to validate a specific run.

## Outputs

- **Figures** in `reports/figures/{figures_subdir}/`: test scatter, residuals, performance by temp, temp distributions, feature importance; optionally color–Teff relations.
- **Text report** at `output.report_file`: overall MAE, RMSE, R², within-5/10/20%, and per-bin stats.
- **JSON report** (same base path, `.json`): same metrics for scripting.

## Troubleshooting

| Issue | Check |
|-------|--------|
| Model not found | `ls models/rf_*.pkl`; ensure `model_pattern` matches. |
| Test predictions not found | Test predictions are created at **training** time. Retrain with the same pipeline so `*_test_predictions.parquet` exists in `models/`. |
| Wrong column names | Predictions file must follow the names expected by the validation pipeline (e.g. predicted vs true columns). |

## See also

- `config/models/README.md` – Training configs (training produces the test predictions used here).
- `src/pipeline/validation_pipeline.py` – Validation pipeline implementation.
- `src/visualization/validation_plots.py` – Plotting functions.
