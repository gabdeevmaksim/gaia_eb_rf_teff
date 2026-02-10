# Configurable ML Training Pipeline

## Overview

The configurable ML pipeline eliminates code duplication by using YAML configuration files to define model variants. Instead of creating a separate training script for each model, you now use **one pipeline with different configuration files**.

## Problem Solved

**Before** (duplicated code):
- 10+ separate training scripts (`train_gaia_2mass_ir_model.py`, `train_gaia_g_bprp_engineered_model.py`, etc.)
- 200-400 lines of duplicated boilerplate per script
- Bug fixes needed in multiple places
- Hard to maintain consistency

**After** (configurable pipeline):
- 1 universal pipeline (`ConfigurableMLPipeline`)
- 5 YAML config files (one per model variant)
- Single source of truth for training logic
- Easy to create new model variants

---

## Quick Start

### Train a Model

```bash
# Activate environment
source .venv/bin/activate

# Train using a model configuration
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log_optuna.yaml

# Dry run to see what would execute
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log_optuna.yaml --dry-run
```

### Available Model Configurations

Training data is read from `data/raw/` when `source_location: raw` (e.g. `eb_unified_photometry.parquet`). See `config/models/README.md`.

```bash
config/models/
├── gaia_teff_corrected_log_optuna.yaml      # Teff from Gaia colors (log, corrected)
├── gaia_teff_corrected_clustering_optuna.yaml  # Teff with clustering features
├── gaia_teff_flag1_corrected_log_optuna.yaml   # Teff flag1-corrected
├── gaia_logg_optuna.yaml                    # logg from Gaia colors
├── gaia_logg_teff_corrected_log_optuna.yaml # Teff from Gaia + logg (chain)
├── template_training_config.yaml            # Template for new models
└── model_registry.yaml                      # HuggingFace model registry
```

---

## Pipeline Architecture

```
ConfigurableMLPipeline
│
├── 1. Load Model Configuration (YAML)
├── 2. Load Training Data (from config)
├── 3. Apply Teff Correction (if enabled) **NEW**
├── 4. Preprocess Data (filter missing values)
├── 5. Engineer Features (if enabled)
├── 6. Prepare Train/Test Split
├── 7. Train Random Forest Model
├── 8. Evaluate Performance
└── 9. Save Model + Artifacts
```

Each step reads from the shared configuration, making the pipeline adapt to different model variants automatically.

---

## Configuration Schema

### Complete Example

```yaml
# config/models/example.yaml

model:
  name: "Human-readable model name"
  id_prefix: "file_prefix_for_saved_models"
  description: "Detailed description of what this model does"

data:
  # Data source
  source_file: "eb_unified_photometry.parquet"
  source_location: "raw"   # or "processed" for files in data/processed

  # Target variable column
  target: "teff_gaia_corrected"

  # Columns to exclude from features
  exclude_columns:
    - "source_id"
    - "other_metadata"

  # Feature columns (null = auto-detect all except target/excluded)
  features: null
  # OR specify explicitly:
  # features:
  #   - "bp_rp"
  #   - "j_h_color"

feature_engineering:
  # Apply feature engineering during pipeline?
  enabled: false

  # Colors to engineer (if enabled=true)
  color_cols:
    - "g_r_color"
    - "bp_rp"

  # Magnitudes to engineer (if enabled=true)
  mag_cols:
    - "gPSFMag"

preprocessing:
  # Filter missing values?
  filter_missing: true
  missing_value: -999.0

  # Use sample weights for training?
  use_sample_weights: false
  weight_column: "sample_weight"

# Teff Correction (NEW - optional)
# Applies polynomial correction to hot stars before training
teff_correction:
  enabled: true                                    # Enable/disable correction
  target_column: "teff_gaia"                       # Original Teff column
  threshold: 10000                                 # Apply correction for Teff > 10,000 K
  coefficients_file: "teff_correction_coeffs_deg2.pkl"  # Polynomial coefficients
  # Note: Creates new column '{target_column}_corrected'

hyperparameters:
  n_estimators: 300
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 4
  max_features: "log2"  # Can be "sqrt", "log2", or integer
  n_jobs: -1            # Use all CPU cores
  random_state: 42

training:
  test_size: 0.2
  random_state: 42

validation:
  create_plots: true    # Future: auto-generate validation plots
  n_temp_bins: 5
```

### Configuration Sections

#### 1. `model` (required)

Defines model metadata:
- `name`: Display name for logs and reports
- `id_prefix`: Prefix for saved model files (e.g., `rf_gaia_2mass_ir`)
- `description`: What this model predicts and how

#### 2. `data` (required)

Data loading configuration:
- `source_file`: Parquet or CSV file in `data/processed/`
- `target`: Target variable column name
- `exclude_columns`: Metadata columns to exclude from features
- `features`: `null` (auto-detect) or explicit list

**Auto-detection**: If `features: null`, uses all columns except `target` and `exclude_columns`.

#### 3. `feature_engineering` (optional)

Controls dynamic feature engineering:
- `enabled: true`: Apply `engineer_all_features()` during pipeline
- `enabled: false`: Use pre-engineered features from dataset
- `color_cols`: Colors to engineer (if enabled)
- `mag_cols`: Magnitudes to engineer (if enabled)

**When to use:**
- `enabled: true`: Dataset has raw colors/magnitudes, need polynomial/interaction features
- `enabled: false`: Dataset already has engineered features

#### 4. `preprocessing` (optional)

Data preprocessing:
- `filter_missing`: Remove rows with missing values?
- `missing_value`: Value indicating missing data (default: -999.0)
- `use_sample_weights`: Apply distribution matching weights?
- `weight_column`: Column name for weights

#### 5. `teff_correction` (optional) **NEW**

Polynomial correction for hot stars (Teff > threshold):
- `enabled`: Apply correction during pipeline (default: false)
- `target_column`: Column containing original Teff values
- `threshold`: Apply correction for stars with Teff > threshold (Kelvin)
- `coefficients_file`: Path to joblib file with polynomial coefficients (relative to data root)

**What it does:**
- Loads polynomial coefficients from specified file
- Applies correction: `Teff_corrected = c0 + c1*Teff + c2*Teff^2 + ...`
- Creates new column: `{target_column}_corrected`
- Logs correction statistics (mean/median correction, number of stars affected)
- Saves correction info in model metadata

**When to use:**
- Gaia GSP-Phot systematically underestimates Teff for hot stars (>10,000K)
- Use this to train models on corrected temperatures
- Update `data.target` to use the corrected column: `teff_gaia_corrected`

**Example:**
```yaml
teff_correction:
  enabled: true
  target_column: "teff_gaia"
  threshold: 10000
  coefficients_file: "teff_correction_coeffs_deg2.pkl"

data:
  target: "teff_gaia_corrected"  # Use corrected values as target
```

#### 6. `hyperparameters` (optional)

Random Forest parameters (defaults from `config/config.yaml` if omitted):
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples per leaf
- `max_features`: Features per split ("sqrt", "log2", or number)
- `n_jobs`: Parallel jobs (-1 = all CPUs)
- `random_state`: Reproducibility seed

#### 7. `training` (optional)

Train/test split configuration:
- `test_size`: Fraction for test set (0.0-1.0)
- `random_state`: Split reproducibility seed

#### 8. `validation` (optional)

Validation configuration (future use):
- `create_plots`: Generate validation plots automatically
- `n_temp_bins`: Temperature bins for performance analysis

---

## Usage Examples

### Example 1: Train Gaia + 2MASS Model

```bash
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml
```

**Output:**
```
Model saved: rf_gaia_2mass_ir_20251103_141119
Test MAE: 589 K
Test R²: 0.593
```

**Created files:**
```
models/rf_gaia_2mass_ir_20251103_141119.pkl          # Trained model
models/rf_gaia_2mass_ir_20251103_141119_metadata.json   # Configuration
models/rf_gaia_2mass_ir_20251103_141119_SUMMARY.txt     # Performance summary
models/rf_gaia_2mass_ir_20251103_141119_test_predictions.parquet  # Test predictions
```

### Example 2: Create Custom Model Config

```bash
# 1. Copy template
cp config/models/gaia_2mass_ir.yaml config/models/my_new_model.yaml

# 2. Edit configuration
nano config/models/my_new_model.yaml

# 3. Update fields:
model:
  name: "My Custom Model"
  id_prefix: "rf_custom"
  description: "Custom temperature model"

data:
  source_file: "my_training_data.parquet"
  target: "teff_gspphot"
  features:
    - "feature1"
    - "feature2"

# 4. Validate with dry run
python pipeline.py --ml --ml-config config/models/my_new_model.yaml --dry-run

# 5. Train model
python pipeline.py --ml --ml-config config/models/my_new_model.yaml
```

### Example 3: Hyperparameter Tuning

Create multiple configs with different hyperparameters:

**config/models/gaia_2mass_deep.yaml:**
```yaml
model:
  id_prefix: "rf_gaia_2mass_deep"

hyperparameters:
  n_estimators: 500  # More trees
  max_depth: 30      # Deeper trees
```

**config/models/gaia_2mass_wide.yaml:**
```yaml
model:
  id_prefix: "rf_gaia_2mass_wide"

hyperparameters:
  n_estimators: 1000  # Many shallow trees
  max_depth: 10
```

Train both and compare:
```bash
python pipeline.py --ml --ml-config config/models/gaia_2mass_deep.yaml
python pipeline.py --ml --ml-config config/models/gaia_2mass_wide.yaml
```

---

## Programmatic Usage

Use the pipeline in Python scripts or notebooks:

```python
from src.pipeline import ConfigurableMLPipeline

# Train model
pipeline = ConfigurableMLPipeline('config/models/gaia_2mass_ir.yaml')
context = pipeline.run()

# Access results
print(f"Model ID: {context['model_id']}")
print(f"Test MAE: {context['test_metrics']['mae']:.0f} K")
print(f"Test R²: {context['test_metrics']['r2']:.4f}")

# Model saved at
print(f"Model file: {context['model_file']}")
```

### Batch Training

Train multiple models programmatically:

```python
from pathlib import Path
from src.pipeline import ConfigurableMLPipeline

# Get all model configs
config_dir = Path('config/models')
configs = list(config_dir.glob('*.yaml'))

results = []

for config_file in configs:
    print(f"\nTraining {config_file.stem}...")

    try:
        pipeline = ConfigurableMLPipeline(str(config_file))
        context = pipeline.run()

        results.append({
            'config': config_file.stem,
            'model_id': context['model_id'],
            'test_mae': context['test_metrics']['mae'],
            'test_r2': context['test_metrics']['r2']
        })
    except Exception as e:
        print(f"Failed: {e}")

# Compare models
import pandas as pd
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('test_mae')
print("\nModel Comparison:")
print(df_results)
```

---

## Benefits

### 1. No Code Duplication
- Single pipeline implementation
- Fix bugs once, applies everywhere
- Consistent behavior across all models

### 2. Easy Experimentation
- Change hyperparameters in YAML, no code changes
- Quick A/B testing of model variants
- Clear documentation of what makes each model different

### 3. Reproducibility
- Configuration files version-controlled with code
- Model metadata includes exact config used
- Easy to recreate any model from its config

### 4. Maintainability
- New model = new YAML file, not new script
- Centralized logic easier to understand
- Configuration schema validates correctness

### 5. Scalability
- Easy to automate training runs
- Parallel training of multiple models
- CI/CD integration ready

---

## Migration from Legacy Scripts

### Old Approach

```bash
# 10 separate scripts
python scripts/train_gaia_2mass_ir_model.py
python scripts/train_gaia_g_bprp_model.py
python scripts/train_gaia_g_bprp_engineered_model.py
python scripts/train_panstarrs_basic_model.py
# ... 6 more scripts
```

Each script: 200-400 lines of similar code

### New Approach

```bash
# 1 pipeline, 5 configs
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml
python pipeline.py --ml --ml-config config/models/gaia_g_bprp.yaml
python pipeline.py --ml --ml-config config/models/gaia_g_bprp_engineered.yaml
python pipeline.py --ml --ml-config config/models/panstarrs_basic.yaml
# ... use different configs
```

**Result:**
- ~3,000 lines of duplicate code → 540 lines reusable pipeline
- 10 scripts → 5 configs
- Better: 1 source of truth

---

## Comparison: Old vs New

| Aspect | Old (Separate Scripts) | New (Configurable Pipeline) |
|--------|------------------------|----------------------------|
| **Code lines** | ~3,000 (10 scripts × 300 lines) | ~540 (1 pipeline) |
| **Add new model** | Copy/modify 300-line script | Create 50-line YAML |
| **Change hyperparam** | Edit Python code | Edit YAML value |
| **Fix bug** | Update 10 scripts | Update 1 pipeline |
| **Consistency** | Manual, error-prone | Automatic, guaranteed |
| **Documentation** | In docstrings | In config schema |
| **Testing** | Test each script | Test once |

---

## Troubleshooting

### Error: "Model config not found"

```
FileNotFoundError: Model config not found: config/models/my_model.yaml
```

**Solution:** Check file path is correct and file exists:
```bash
ls -l config/models/my_model.yaml
```

### Error: "Training data not found"

```
FileNotFoundError: Training data not found: data/processed/my_data.parquet
```

**Solution:** Verify `data.source_file` in config and file exists:
```bash
ls -l data/processed/my_data.parquet
```

### Error: "Column not found"

```
ColumnNotFoundError: unable to find column "Te_avg"
```

**Solution:** Check `data.target` matches actual column name in dataset:
```python
import polars as pl
df = pl.read_parquet('data/processed/your_data.parquet')
print(df.columns)  # Find correct column name
```

### Error: "Input contains infinity"

```
ValueError: Input X contains infinity or a value too large
```

**Solution:** Feature engineering created inf values. Options:
1. Disable feature engineering: `feature_engineering.enabled: false`
2. Use pre-engineered dataset without inf
3. Add inf/nan handling to preprocessing

### Slow Training

Pipeline taking too long?

**Solutions:**
1. Reduce `n_estimators` (e.g., 300 → 100)
2. Reduce `max_depth` (e.g., 20 → 15)
3. Use smaller dataset for testing
4. Ensure `n_jobs: -1` (use all CPUs)

---

## Next Steps

1. **Create more model configs**: See `config/models/README.md` for examples
2. **Automate training**: Use bash script or cron jobs
3. **Compare models**: Use programmatic approach to train and compare
4. **Extend pipeline**: Add custom steps for your specific needs

---

## Technical Details

### Pipeline Steps

Each pipeline step:
- Receives `context` dict with previous outputs
- Performs its operation
- Returns updated `context`
- Logs timing and status

### Context Flow

```python
context = {
    'config': Config(),                    # Project config
    'model_config': {...},                 # Model YAML config
    'raw_data': DataFrame,                 # Loaded data
    'clean_data': DataFrame,               # After preprocessing
    'feature_data': DataFrame,             # After feature engineering
    'X_train': DataFrame,                  # Training features
    'X_test': DataFrame,                   # Test features
    'y_train': Series,                     # Training target
    'y_test': Series,                      # Test target
    'model': RandomForestRegressor,        # Trained model
    'train_metrics': {...},                # Training performance
    'test_metrics': {...},                 # Test performance
    'feature_importance': {...},           # Feature importances
    'model_id': 'rf_model_20251103_141119',# Saved model ID
}
```

### Adding Custom Steps

Extend the pipeline with custom steps:

```python
from src.pipeline.base import PipelineStep
from src.pipeline.configurable_ml_pipeline import ConfigurableMLPipeline

class MyCustomStep(PipelineStep):
    def __init__(self):
        super().__init__("My Custom Step")

    def run(self, context):
        # Your logic here
        self.logger.info("Running custom logic...")
        context['my_output'] = ...
        return context

# Create custom pipeline
class MyCustomPipeline(ConfigurableMLPipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        # Insert custom step
        self.steps.insert(6, MyCustomStep())  # After training
```

---

## See Also

- `config/models/README.md` - Model configuration examples
- `docs/PIPELINES.md` - Original pipeline documentation
- `docs/CONFIGURATION.md` - Project configuration system
- `src/pipeline/configurable_ml_pipeline.py` - Implementation

---

## Summary

The configurable ML pipeline provides a production-ready, maintainable solution for training temperature prediction models:

✅ **One pipeline, many models** - No code duplication
✅ **YAML configuration** - Easy to modify without code changes
✅ **Automatic logging** - Track every step with timing
✅ **Reproducible** - Config files document exact setup
✅ **Extensible** - Add custom steps easily
✅ **Tested** - Successfully trained Gaia+2MASS model (MAE: 589K, R²: 0.593)

**Start using it today:**
```bash
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml
```
