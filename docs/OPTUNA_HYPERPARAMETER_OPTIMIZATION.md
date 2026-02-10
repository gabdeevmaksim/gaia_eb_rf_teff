# Optuna Hyperparameter Optimization

## Overview

The ML training pipeline now includes automatic hyperparameter optimization using Optuna. This feature:

1. **Automatically finds optimal hyperparameters** for your model configuration
2. **Caches optimized hyperparameters** in the config file for reuse
3. **Skips optimization** if hyperparameters already exist for the same model settings
4. **Saves time** by avoiding redundant optimization runs

## How It Works

### Model Signature

The pipeline generates a unique "signature" for each model configuration based on:
- Data source file
- Target column
- Features used
- Preprocessing settings (filters, missing values)
- Feature engineering settings
- Clustering settings
- Target transformation
- Training settings (test_size, random_state)

This signature ensures that hyperparameters are only reused when the model configuration is truly identical.

### Optimization Flow

1. **Generate signature** from model configuration
2. **Check cache** for existing hyperparameters with this signature
3. **If found**: Use cached hyperparameters (skip optimization)
4. **If not found**: 
   - Run Optuna optimization (default: 50 trials)
   - Save optimized hyperparameters to config file
   - Use optimized hyperparameters for training

### Hyperparameter Cache

Optimized hyperparameters are stored in the config file under `hyperparameter_cache`:

```yaml
hyperparameter_cache:
  abc123def456:  # Model signature hash
    hyperparameters:
      n_estimators: 450
      max_depth: 25
      min_samples_split: 3
      min_samples_leaf: 2
      max_features: sqrt
      random_state: 42
      n_jobs: -1
    optimized_at: "2026-01-26T10:30:00"
    n_trials: 50
```

The main `hyperparameters` section is also updated with the optimized values.

## Usage

### Enable Optimization

Add the `optuna_optimization` section to your model config file:

```yaml
# Optuna Hyperparameter Optimization
optuna_optimization:
  enabled: true                    # Enable Optuna optimization
  n_trials: 50                     # Number of optimization trials (default: 50)
  timeout: null                    # Max time in seconds (null = no limit)
```

### Disable Optimization

Set `enabled: false` to use hyperparameters directly from the config:

```yaml
optuna_optimization:
  enabled: false  # Use hyperparameters from config directly
```

### Example Config

```yaml
model:
  name: "Gaia Teff Model"
  id_prefix: "rf_gaia_teff"
  description: "Temperature prediction using Gaia colors"

data:
  source_file: "eb_unified_photometry.parquet"
  target: "teff_gaia"
  features:
    - "bp_rp"
    - "g_bp"
    - "g_rp"

training:
  test_size: 0.2
  random_state: 42

# Enable Optuna optimization
optuna_optimization:
  enabled: true
  n_trials: 50
  timeout: null

# These will be updated by Optuna if optimization is enabled
hyperparameters:
  n_estimators: 300      # Default, will be optimized
  max_depth: 20          # Default, will be optimized
  min_samples_split: 5   # Default, will be optimized
  min_samples_leaf: 2    # Default, will be optimized
  max_features: sqrt     # Default, will be optimized
  random_state: 42
  n_jobs: -1
```

## Optimization Parameters

Optuna optimizes the following Random Forest hyperparameters:

- **n_estimators**: 100-1000 (step: 50)
- **max_depth**: 5-50
- **min_samples_split**: 2-20
- **min_samples_leaf**: 1-10
- **max_features**: 'sqrt', 'log2', or None

The optimization uses **Mean Absolute Error (MAE)** as the objective function, minimizing validation error.

## Benefits

1. **Automatic optimization**: No need to manually tune hyperparameters
2. **Caching**: Hyperparameters are saved and reused automatically
3. **Reproducibility**: Same model settings always use the same hyperparameters
4. **Time savings**: Skip optimization on subsequent runs with same settings
5. **Better performance**: Optimized hyperparameters typically improve model accuracy

## Installation

Optuna is automatically included in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Or install Optuna separately:

```bash
pip install optuna
```

## Troubleshooting

### Optuna Not Installed

If you see an error about Optuna not being installed:

```bash
pip install optuna
```

### Optimization Takes Too Long

Reduce the number of trials or set a timeout:

```yaml
optuna_optimization:
  enabled: true
  n_trials: 20        # Reduce from 50
  timeout: 3600       # 1 hour limit
```

### Want to Re-optimize

To force re-optimization, delete the `hyperparameter_cache` section from your config file, or remove the specific signature entry.

## Technical Details

### Signature Generation

The model signature is a SHA256 hash of:
- Data source file path
- Target column name
- Sorted list of features
- Preprocessing filters (sorted)
- Feature engineering settings
- Clustering settings
- Target transformation
- Training parameters

This ensures that hyperparameters are only reused when the model configuration is truly identical.

### Optimization Strategy

- Uses **TPE (Tree-structured Parzen Estimator)** sampler (Optuna default)
- Optimizes on a 20% validation split of training data
- Supports sample weights if provided
- Minimizes Mean Absolute Error (MAE)

### Cache Location

Hyperparameters are cached in the same config file that defines the model, under the `hyperparameter_cache` key. This keeps everything in one place and makes it easy to version control optimized hyperparameters.
