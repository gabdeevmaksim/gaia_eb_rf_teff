# Model Training Configuration Template Guide

## Overview

The `config/models/template_training_config.yaml` file contains a comprehensive template with **all possible configuration options** for the ConfigurableMLPipeline. This guide explains each section and how to use them.

## Quick Start

1. Copy the template:
   ```bash
   cp config/models/template_training_config.yaml config/models/my_model.yaml
   ```

2. Edit `my_model.yaml` with your settings

3. Train the model:
   ```bash
   python pipeline.py --ml-config config/models/my_model.yaml
   ```

## Configuration Sections

### 1. Model Metadata (`model:`)
**Required** - Basic information about your model
- `name`: Display name
- `id_prefix`: Prefix for model ID (e.g., `rf_gaia_teff`)
- `description`: What the model does

### 2. Teff Correction (`teff_correction:`)
**Optional** - Apply polynomial correction to hot stars
- `enabled`: true/false
- `target_column`: Original Teff column to correct
- `threshold`: Apply correction for Teff > threshold
- `coefficients_file`: Polynomial coefficients file

**Note**: Creates a new column `{target_column}_corrected` - use this as your target!

### 3. Target Transformation (`target_transform:`)
**Optional** - Transform target variable
- Options: `none`, `log`, `log2`, `ln`
- Common for temperature: use `log` (log10)

### 4. Data Configuration (`data:`)
**Required** - Data source and features
- `source_file`: Dataset filename
- `source_location`: Where to find file (raw/processed/external/interim)
- `target`: Target column name
- `features`: List of feature columns (empty = auto-detect all)
- `exclude_columns`: Columns to exclude from auto-detection
- `id_column`: Unique identifier column

### 5. Preprocessing (`preprocessing:`)
**Optional** - Data cleaning and filtering
- `missing_value`: Missing value indicator (default: -999.0)
- `filters`: Value range filters `{column: [min, max]}`
- `drop_missing`: Drop rows with missing values (default: true)
- `use_sample_weights`: Enable sample weights
- `weight_column`: Column with sample weights

### 6. Feature Engineering (`feature_engineering:`)
**Optional** - Create additional features
- `enabled`: true/false
- `color_cols`: Columns for polynomial/interaction features
- `mag_cols`: Magnitude columns
- `include_polynomials`: Create x^2, x^3, etc.
- `include_interactions`: Create x1 * x2
- `include_log`: Create log(x)
- `include_temp_dependent`: Temperature-dependent features
- `include_mag_features`: Magnitude-based features
- `custom_features`: Custom pandas eval expressions

### 7. Training Configuration (`training:`)
**Optional** - Train/test split settings
- `test_size`: Fraction for test set (0.0-1.0, default: 0.2)
- `random_state`: Random seed (default: 42)

### 8. Optuna Optimization (`optuna_optimization:`)
**Optional** - Automatic hyperparameter optimization
- `enabled`: true/false
- `n_trials`: Number of optimization trials (default: 50)
- `timeout`: Max time in seconds (null = no limit)

**How it works:**
1. Generates unique signature from config
2. Checks for cached hyperparameters
3. If found, uses cached (skips optimization)
4. If not found, runs Optuna and saves results
5. Subsequent runs reuse cached hyperparameters

### 9. Hyperparameters (`hyperparameters:`)
**Required** - Random Forest parameters
- `n_estimators`: Number of trees
- `max_depth`: Max tree depth (null = unlimited)
- `min_samples_split`: Min samples to split
- `min_samples_leaf`: Min samples at leaf
- `max_features`: Features per split ("sqrt", "log2", int, null)
- `random_state`: Random seed
- `n_jobs`: Parallel jobs (-1 = all cores)

**Note**: If Optuna is enabled, these will be updated automatically.

### 10. Clustering Features (`clustering:`)
**Optional** - Add cluster probabilities as features
- `enabled`: true/false
- `method`: `gmm`, `bayesian_gmm`, or `kmeans`
- `n_clusters`: Number of clusters
- `covariance_type`: For GMM (full/tied/diag/spherical)
- `n_init`: Number of initializations
- `max_iter`: Maximum iterations
- `reg_covar`: Covariance regularization

**Note**: Adds `n_clusters` new features (cluster probabilities)

### 11. Output Configuration (`output:`)
**Optional** - What to save
- `save_model`: Save model file (default: true)
- `save_predictions`: Save predictions (default: true)
- `save_test_predictions`: Save test predictions (default: true)
- `models_dir`: Directory for models (default: "models")
- `predictions_dir`: Directory for predictions

### 12. Validation (`validation:`)
**Optional** - Validation plots and analysis
- `create_plots`: Generate plots (default: true)
- `n_temp_bins`: Temperature bins for analysis
- `target_info`: Display information for plots
  - `name`: Display name
  - `unit`: Unit (e.g., "K")
  - `short`: Short name

### 13. Hyperparameter Cache (`hyperparameter_cache:`)
**Auto-generated** - Stores optimized hyperparameters
- Do not edit manually
- Created automatically by Optuna
- Enables reuse of optimized hyperparameters

## Common Patterns

### Basic Model (No Feature Engineering)
```yaml
feature_engineering:
  enabled: false

clustering:
  enabled: false

optuna_optimization:
  enabled: false
```

### Model with Feature Engineering
```yaml
feature_engineering:
  enabled: true
  color_cols: ["bp_rp", "g_bp", "g_rp"]
  include_polynomials: true
  include_interactions: true
```

### Model with Optuna Optimization
```yaml
optuna_optimization:
  enabled: true
  n_trials: 100
  timeout: 3600  # 1 hour
```

### Model with Clustering Features
```yaml
clustering:
  enabled: true
  method: "gmm"
  n_clusters: 5
```

### Model with Teff Correction
```yaml
teff_correction:
  enabled: true
  target_column: "teff_gaia"
  threshold: 10000

data:
  target: "teff_gaia_corrected"  # Use corrected column!
```

### Model with Sample Weights
```yaml
preprocessing:
  use_sample_weights: true
  weight_column: "sample_weight"
```

## Pipeline Execution Order

The pipeline executes in this order:

1. **Load Model Configuration** - Reads this YAML file
2. **Load Training Data** - Loads dataset from `data.source_file`
3. **Apply Teff Correction** - If enabled, creates corrected column
4. **Preprocess Data** - Applies filters, handles missing values
5. **Engineer Features** - If enabled, creates polynomial/interaction features
6. **Prepare Train/Test Split** - Splits data, applies target transformation
7. **Add Clustering Features** - If enabled, adds cluster probabilities
8. **Optimize Hyperparameters** - If enabled, runs Optuna optimization
9. **Train Model** - Trains Random Forest with hyperparameters
10. **Evaluate Model** - Calculates metrics on train and test sets
11. **Save Model** - Saves model, metadata, predictions, and plots

## Tips

1. **Start Simple**: Begin with basic config, then add features incrementally
2. **Use Optuna**: Enable it for better hyperparameters (cached automatically)
3. **Feature Engineering**: Can create many features - test on small dataset first
4. **Missing Values**: Automatically handled (-999.0 → NaN)
5. **Target Transform**: Use `log` for temperature (log10)
6. **Reproducibility**: Set `random_state` in both `training` and `hyperparameters`

## Troubleshooting

### "Dataset not found"
- Check `source_file` and `source_location` paths
- Verify file exists in the specified location

### "Target column not found"
- If using Teff correction, use the corrected column name
- Check column name matches exactly (case-sensitive)

### "Feature column not found"
- Verify all feature columns exist in dataset
- Check for typos in column names

### Optuna takes too long
- Reduce `n_trials`
- Set `timeout` to limit optimization time
- Disable Optuna and use manual hyperparameters

### Too many features
- Disable feature engineering
- Reduce `color_cols` list
- Use `exclude_columns` to filter unwanted columns
