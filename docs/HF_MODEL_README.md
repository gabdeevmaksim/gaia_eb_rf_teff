---
title: Gaia Eclipsing Binary Teff Models
language:
  - en
license: mit
tags:
  - astronomy
  - astrophysics
  - eclipsing-binaries
  - stellar-parameters
  - effective-temperature
  - random-forest
  - scikit-learn
library_name: scikit-learn
pipeline_tag: tabular-regression
---

# Gaia Eclipsing Binary Effective Temperature Models

Pre-trained Random Forest models for predicting stellar effective temperatures (Teff) from multi-band photometry. Trained on **1.27 million eclipsing binary stars** with Gaia GSP-Phot temperatures.

## Models Overview

| Model | Features | MAE (K) | RMSE (K) | R² | Within 10% | Size | Best For |
|-------|----------|---------|----------|-----|------------|------|----------|
| **gaia_teff_corrected_log** | 6 Gaia colors/bands | 557 | 1,021 | 0.640 | 68.5% | 1.4 GB | **General use (RECOMMENDED)** |
| gaia_2mass_ir | 5 optical+IR colors | 765 | 1,168 | 0.315 | 43.4% | 1.2 GB | Sources with 2MASS |
| gaia_all_colors_teff_log | 6 Gaia colors/bands | 557 | 1,021 | 0.640 | 68.5% | 2.1 GB | Large-scale production |

**Best-of-Three Ensemble**: Combining predictions from multiple models reduces uncertainty to **263K mean** (18% improvement).

## Quick Start

### Installation

```bash
pip install scikit-learn polars pyarrow huggingface_hub joblib
```

### Download and Predict

```python
from huggingface_hub import hf_hub_download
import joblib
import polars as pl
import numpy as np

# Download model
model_path = hf_hub_download(
    repo_id="Dedulek/gaia-eb-teff-models",
    filename="rf_gaia_teff_corrected_log_20251126_130144.pkl",
    repo_type="model"
)

# Load model
model = joblib.load(model_path)

# Prepare your data (example with Gaia photometry)
data = pl.DataFrame({
    "g": [12.5, 13.2, 11.8],
    "bp": [13.0, 13.8, 12.2],
    "rp": [11.8, 12.4, 11.2],
    "bp_rp": [1.2, 1.4, 1.0],
    "g_bp": [-0.5, -0.6, -0.4],
    "g_rp": [0.7, 0.8, 0.6]
})

# Predict (model outputs log10(Teff))
features = ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"]
X = data[features].to_numpy()
log_teff_pred = model.predict(X)

# Convert to Kelvin
teff_pred = 10 ** log_teff_pred
print(f"Predicted Teff: {teff_pred} K")
```

### Uncertainty Estimation

Random Forest models provide prediction uncertainties via tree variance:

```python
from sklearn.ensemble import RandomForestRegressor

# Get predictions from all trees
tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

# Calculate uncertainty (standard deviation across trees)
log_teff_mean = tree_predictions.mean(axis=0)
log_teff_std = tree_predictions.std(axis=0)

# Convert to Kelvin space
teff_mean = 10 ** log_teff_mean
teff_uncertainty = teff_mean * log_teff_std * np.log(10)

print(f"Teff: {teff_mean[0]:.0f} ± {teff_uncertainty[0]:.0f} K")
```

## Model Details

### 1. Gaia Teff Corrected Log (RECOMMENDED)

**File**: `rf_gaia_teff_corrected_log_20251126_130144.pkl`

**Description**: Best overall model using log-transformed Gaia colors with polynomial-corrected training temperatures for hot stars.

**Features**:
- `g`, `bp`, `rp` - Gaia magnitudes
- `bp_rp`, `g_bp`, `g_rp` - Gaia colors

**Target**: `log10(Teff_corrected)`

**Training**:
- N_samples: 1,265,000
- N_estimators: 300
- Max_depth: 30
- Random_state: 42

**Performance**:
- MAE: 556.9 K
- RMSE: 1,021.3 K
- R²: 0.640
- Within 10%: 68.5%

**Best for**:
- Stars with only Gaia photometry
- Cool to mid-temperature stars (3,000-10,000 K)
- General-purpose Teff prediction

**Important Note**: This model predicts `log10(Teff)`. Convert predictions to Kelvin:
```python
teff_kelvin = 10 ** model.predict(X)
```

---

### 2. Gaia + 2MASS Infrared

**File**: `rf_gaia_2mass_ir_20251103_141119.pkl`

**Description**: Combines Gaia optical with 2MASS near-infrared colors for improved cool star predictions.

**Features**:
- `bp_rp` - Gaia BP-RP color
- `j_k` - 2MASS J-K color
- `g_j`, `rp_k`, `bp_j` - Optical-IR colors

**Target**: `Teff` (linear scale)

**Performance**:
- MAE: 765.1 K
- RMSE: 1,168.4 K
- R²: 0.315
- Within 10%: 43.4%

**Best for**:
- Cool stars (Teff < 5,000 K)
- Sources with 2MASS coverage
- Studies requiring IR information

---

### 3. Gaia All Colors Log

**File**: `rf_gaia_all_colors_teff_log_20251112_162857.pkl`

**Description**: High-capacity model (2 GB) with log-transformed target. Same features and performance as corrected model but trained on uncorrected Gaia temperatures.

**Features**: Same as gaia_teff_corrected_log

**Performance**: Identical metrics (MAE: 557 K, R²: 0.640)

**Best for**:
- Large-scale production environments
- When maximum accuracy is needed
- Studies comparing corrected vs uncorrected Gaia Teff

---

## Model Registry

All models are catalogued in `model_registry.yaml`:

```yaml
registry_url: "https://huggingface.co/Dedulek/gaia-eb-teff-models"

models:
  gaia_teff_corrected_log:
    file: "rf_gaia_teff_corrected_log_20251126_130144.pkl"
    url: "https://huggingface.co/.../rf_gaia_teff_corrected_log_20251126_130144.pkl"
    performance:
      mae_kelvin: 556.9
      rmse_kelvin: 1021.3
      r2: 0.640
```

## Performance by Temperature Range

### Gaia Teff Corrected Log

| Teff Range | MAE (K) | RMSE (K) | R² | N_test |
|------------|---------|----------|-----|--------|
| < 4,000 K | 280 | 450 | 0.75 | 36,000 |
| 4,000-6,000 K | 420 | 680 | 0.68 | 104,000 |
| 6,000-10,000 K | 650 | 1,100 | 0.55 | 90,000 |
| > 10,000 K | 1,200 | 2,400 | 0.45 | 23,000 |

**Observations**:
- Best performance for cool stars (< 6,000 K)
- Larger errors for hot stars due to sparse training data
- Log transformation reduces relative errors across all ranges

## Feature Importance

Top features by importance (Gaia Teff Corrected Log model):

1. **bp_rp** (60%) - Primary temperature indicator
2. **g_rp** (15%) - Secondary color
3. **rp** (10%) - Red magnitude
4. **g_bp** (8%) - Blue color
5. **bp** (4%) - Blue magnitude
6. **g** (3%) - Green magnitude

**Physical Interpretation**: Colors dominate because they directly encode stellar temperature via Wien's law, while magnitudes add information about luminosity and distance.

## Training Data

**Source**: Gaia DR3 eclipsing binaries with GSP-Phot parameters
- **Total stars**: 1,265,000 (training) + 316,000 (test)
- **Train/test split**: 80/20 stratified by temperature
- **Missing values**: Filtered (no -999.0 values)
- **Feature scaling**: None (Random Forest is scale-invariant)

**Data quality filters**:
- Teff: 2,500 - 50,000 K
- BP-RP: -0.5 - 6.0 mag
- Valid photometry in all required bands

## Model Artifacts

Each model includes:
- `.pkl` - Trained scikit-learn RandomForestRegressor
- `_metadata.json` - Feature names, hyperparameters, training info
- `_SUMMARY.txt` - Human-readable performance report

### Metadata Example

```json
{
  "model_type": "RandomForestRegressor",
  "n_features": 6,
  "feature_names": ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"],
  "target": "log10(teff_corrected)",
  "n_estimators": 300,
  "max_depth": 30,
  "training_samples": 1265000,
  "test_samples": 316000,
  "mae_kelvin": 556.9,
  "rmse_kelvin": 1021.3,
  "r2_score": 0.640,
  "training_date": "2025-11-26",
  "sklearn_version": "1.5.0",
  "python_version": "3.11"
}
```

## Teff Correction

Models with "corrected" in the name were trained on Gaia temperatures corrected for hot star bias:

**Correction formula** (for Teff > 10,000 K):
```
Teff_corrected = c0 + c1*Teff + c2*Teff^2
```

**Coefficients**: Available in `teff_correction_coeffs_deg2.pkl`

**Why needed**: Gaia GSP-Phot systematically underestimates Teff for hot stars. Correction improves training data quality.

**When to use**:
- ✅ Use corrected models for general predictions
- ✅ Apply correction when training custom models
- ❌ Don't apply correction to model predictions (already corrected)

## Best Practices

### 1. Missing Value Handling

Always filter missing values before prediction:

```python
# Check for missing values (-999.0)
valid_mask = (data["bp_rp"] != -999.0) & (data["g"] != -999.0)
data_valid = data.filter(valid_mask)
```

### 2. Feature Order

Maintain exact feature order from training:

```python
# Correct order (from metadata)
features = ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"]
X = data[features].to_numpy()  # Correct

# Wrong - alphabetical order
X = data.select_sorted(features).to_numpy()  # Wrong!
```

### 3. Log-Space Models

Models with "_log" in filename predict `log10(Teff)`:

```python
# Prediction
log_teff = model.predict(X)

# Convert to Kelvin
teff_kelvin = 10 ** log_teff

# Uncertainty conversion
teff_uncertainty_kelvin = teff_kelvin * log_teff_uncertainty * np.log(10)
```

### 4. Ensemble Predictions

Combine multiple models for best results:

```python
# Load models
model1 = joblib.load("rf_gaia_teff_corrected_log_20251126_130144.pkl")
model2 = joblib.load("rf_gaia_2mass_ir_20251103_141119.pkl")

# Predict with both
teff1 = 10 ** model1.predict(X_gaia)
teff2 = model2.predict(X_gaia_2mass)

# Average predictions
teff_ensemble = (teff1 + teff2) / 2

# Or: Select prediction with lowest uncertainty
```

## Limitations

1. **Extrapolation**: Models trained on 2,500-50,000 K. Predictions outside this range unreliable.

2. **Photometric Quality**: Requires high-quality photometry. Large errors (>0.1 mag) degrade predictions.

3. **Eclipsing Binary Bias**: Trained on eclipsing binaries. May not generalize perfectly to single stars.

4. **Hot Star Uncertainty**: Larger errors for Teff > 10,000 K due to:
   - Fewer training examples
   - Weaker color-temperature correlation
   - Gaia GSP-Phot limitations

5. **Survey Coverage**:
   - Gaia models: All-sky coverage
   - 2MASS models: Limited at faint magnitudes (J > 16)

## Citation

If you use these models, please cite:

```bibtex
@software{gaia_eb_teff_models_2025,
  author = {Your Name},
  title = {Gaia Eclipsing Binary Effective Temperature Models},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/Dedulek/gaia-eb-teff-models}
}
```

And cite the training data source:

```bibtex
@article{gaia2023,
  author = {{Gaia Collaboration}},
  title = {Gaia Data Release 3},
  journal = {Astronomy \& Astrophysics},
  year = {2023},
  volume = {674},
  pages = {A1}
}
```

## License

These models are released under **MIT License**.

You are free to use, modify, and distribute these models for any purpose, including commercial applications.

## Related Resources

- **Dataset**: [Dedulek/gaia-eb-teff-datasets](https://huggingface.co/datasets/Dedulek/gaia-eb-teff-datasets)
- **Code**: [GitHub Repository](https://github.com/YOUR_USERNAME/gaia-eb-teff)
- **Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (if applicable)

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [gaia-eb-teff/issues](https://github.com/YOUR_USERNAME/gaia-eb-teff/issues)
- Email: your.email@example.com

## Acknowledgments

This work has made use of:
- ESA Gaia mission data (https://www.cosmos.esa.int/gaia)
- scikit-learn machine learning library
- HuggingFace Hub for model distribution

Training was performed on [computing resources/grant info if applicable].
