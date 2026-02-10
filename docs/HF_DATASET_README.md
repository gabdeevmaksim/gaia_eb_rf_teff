---
title: Gaia Eclipsing Binary Teff Datasets
language:
  - en
license: cc-by-4.0
size_categories:
  - 1M<n<10M
task_categories:
  - tabular-regression
tags:
  - astronomy
  - astrophysics
  - eclipsing-binaries
  - stellar-parameters
  - photometry
  - gaia
  - effective-temperature
pretty_name: Gaia Eclipsing Binary Effective Temperature Datasets
---

# Gaia Eclipsing Binary Effective Temperature Datasets

## Dataset Description

This dataset contains multi-survey photometry and stellar parameters for **2.18 million eclipsing binary stars** from the Gaia mission. It combines data from Gaia DR3, Pan-STARRS DR1, and 2MASS to enable machine learning prediction of effective temperatures (Teff) for stars lacking spectroscopic measurements.

### Dataset Summary

- **Total objects**: 2,179,680 eclipsing binary stars
- **Gaia DR3 coverage**: 100% (all sources have Gaia photometry)
- **Pan-STARRS coverage**: 53.5% (1,166,000 sources)
- **2MASS coverage**: Variable (J, H, K bands)
- **Teff coverage**: 58% have Gaia GSP-Phot temperatures
- **ML predictions**: 38.9% (847,000 stars) have ML-predicted temperatures

### Surveys Included

1. **Gaia DR3** (2023)
   - G, BP, RP magnitudes and colors
   - GSP-Phot effective temperatures
   - Astrometric parameters

2. **Pan-STARRS DR1** (2016)
   - g, r, i, z, y optical magnitudes
   - PSF and Kron photometry

3. **2MASS** (2003)
   - J, H, K near-infrared magnitudes

## Dataset Structure

### Unified Photometry Dataset

**File**: `photometry/eb_unified_photometry.parquet`
**Size**: 227 MB
**Format**: Apache Parquet

This is the primary dataset containing all photometry and stellar parameters.

#### Key Columns

**Identifiers:**
- `source_id` (int64): Gaia DR3 source identifier

**Gaia Photometry:**
- `g`, `bp`, `rp` (float64): Gaia G, BP, RP magnitudes
- `bp_rp`, `g_bp`, `g_rp` (float64): Gaia colors
- `parallax`, `pmra`, `pmdec` (float64): Astrometry

**Gaia Stellar Parameters:**
- `teff_gaia` (float64): GSP-Phot effective temperature [K]
- `logg_gaia` (float64): Surface gravity [log cm/s²]
- `mh_gaia` (float64): Metallicity [Fe/H]

**Pan-STARRS Photometry:**
- `ps_gPSFMag`, `ps_rPSFMag`, `ps_iPSFMag`, `ps_zPSFMag`, `ps_yPSFMag` (float64): PSF magnitudes
- `ps_gKronMag`, `ps_rKronMag`, etc. (float64): Kron magnitudes
- Pan-STARRS colors: `ps_g_r`, `ps_r_i`, etc.

**2MASS Photometry:**
- `j_m`, `h_m`, `k_m` (float64): 2MASS magnitudes
- `j_h`, `h_k`, `j_k` (float64): 2MASS colors

**Missing Values:**
All missing values are encoded as `-999.0` for consistency.

### Final Catalog with Predictions

**File**: `catalogs/stars_types_with_best_predictions.fits`
**Size**: 196 MB
**Format**: FITS binary table

Complete catalog of 2.1M eclipsing binaries with:
- Original Gaia temperatures (where available)
- ML-predicted temperatures (best-of-three ensemble)
- Prediction uncertainties
- Quality flags (A=Gaia, B/C/D=ML by uncertainty, X=none)

**Coverage**: 97.2% of stars have Teff values (58.3% Gaia original + 38.9% ML predictions)

## Usage

### Download with Python

```python
from huggingface_hub import hf_hub_download
import polars as pl

# Download unified photometry
file_path = hf_hub_download(
    repo_id="Dedulek/gaia-eb-teff-datasets",
    filename="photometry/eb_unified_photometry.parquet",
    repo_type="dataset"
)

# Load with Polars (recommended for large datasets)
df = pl.read_parquet(file_path)

# Or with Pandas
import pandas as pd
df = pd.read_parquet(file_path)

print(f"Loaded {len(df)} eclipsing binaries")
print(f"Columns: {df.columns}")
```

### Download with Hugging Face CLI

```bash
# Install CLI
pip install huggingface_hub

# Download specific file
huggingface-cli download Dedulek/gaia-eb-teff-datasets \
    --repo-type dataset \
    --include "photometry/eb_unified_photometry.parquet" \
    --local-dir ./data
```

### Training Example

```python
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pl.read_parquet("eb_unified_photometry.parquet")

# Filter stars with known Teff and Gaia photometry
df_train = df.filter(
    (pl.col("teff_gaia") != -999.0) &
    (pl.col("bp_rp") != -999.0)
)

# Prepare features and target
features = ["g", "bp", "rp", "bp_rp", "g_bp", "g_rp"]
X = df_train[features].to_numpy()
y = df_train["teff_gaia"].to_numpy()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")
```

## Dataset Statistics

### Photometric Coverage

| Survey | Coverage | N_stars |
|--------|----------|---------|
| Gaia DR3 | 100% | 2,179,680 |
| Pan-STARRS DR1 | 53.5% | 1,166,000 |
| 2MASS (J) | ~60% | ~1,300,000 |
| 2MASS (H) | ~60% | ~1,300,000 |
| 2MASS (K) | ~60% | ~1,300,000 |

### Stellar Parameter Coverage

| Parameter | Coverage | Mean | Std | Range |
|-----------|----------|------|-----|-------|
| Teff (Gaia) | 58% | 7,450 K | 3,200 K | 2,500 - 50,000 K |
| log(g) | 56% | 3.8 | 0.5 | 0.5 - 5.5 |
| [Fe/H] | 48% | -0.2 | 0.4 | -2.5 - +0.5 |

### Temperature Distribution

| Teff Range | N_stars | Percentage |
|------------|---------|------------|
| < 4,000 K (Cool) | 180,000 | 14% |
| 4,000-6,000 K (Mid) | 520,000 | 41% |
| 6,000-10,000 K (Hot) | 450,000 | 36% |
| > 10,000 K (Very Hot) | 115,000 | 9% |

## Data Quality

### Missing Value Convention

All surveys use `-999.0` to indicate missing values. Always filter these before analysis:

```python
# Filter valid measurements
df_clean = df.filter(
    (pl.col("bp_rp") != -999.0) &
    (pl.col("teff_gaia") != -999.0)
)
```

### Known Issues

1. **Gaia GSP-Phot Bias**: Systematic underestimation of Teff for hot stars (>10,000 K)
   - Correction coefficients available in model repository
   - See: `data/teff_correction_coeffs_deg2.pkl`

2. **Pan-STARRS Coverage**: Northern hemisphere bias (Dec > -30°)

3. **2MASS Saturation**: Bright stars (J < 6) may be saturated

## Model Performance

Pre-trained models achieve the following performance on held-out test sets:

| Model | Features | MAE (K) | RMSE (K) | R² | Within 10% |
|-------|----------|---------|----------|-----|------------|
| Gaia Colors (Log) | 6 Gaia colors/bands | 557 | 1,021 | 0.640 | 68.5% |
| Gaia + 2MASS | 5 optical+IR colors | 765 | 1,168 | 0.315 | 43.4% |
| Best-of-Three Ensemble | Multiple models | 263 | - | - | - |

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{gaia_eb_teff_2025,
  author = {Your Name},
  title = {Gaia Eclipsing Binary Effective Temperature Datasets},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/Dedulek/gaia-eb-teff-datasets}
}
```

### Data Sources

Please also cite the original surveys:

**Gaia DR3:**
```bibtex
@article{gaia2023,
  author = {{Gaia Collaboration}},
  title = {Gaia Data Release 3},
  journal = {Astronomy & Astrophysics},
  year = {2023},
  volume = {674},
  pages = {A1}
}
```

**Pan-STARRS:**
```bibtex
@article{panstarrs2020,
  author = {Flewelling, H. A. and others},
  title = {The Pan-STARRS1 Database and Data Products},
  journal = {The Astrophysical Journal Supplement Series},
  year = {2020},
  volume = {251},
  pages = {7}
}
```

**2MASS:**
```bibtex
@article{2mass2006,
  author = {Skrutskie, M. F. and others},
  title = {The Two Micron All Sky Survey (2MASS)},
  journal = {The Astronomical Journal},
  year = {2006},
  volume = {131},
  pages = {1163}
}
```

## License

This dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- Share: copy and redistribute the material
- Adapt: remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit and indicate if changes were made

## Contact

For questions or issues with this dataset:
- Open an issue on the [GitHub repository](https://github.com/YOUR_USERNAME/gaia-eb-teff)
- Contact: your.email@example.com

## Acknowledgments

This work has made use of data from:
- ESA mission Gaia (https://www.cosmos.esa.int/gaia)
- Pan-STARRS (https://panstarrs.stsci.edu/)
- 2MASS (https://www.ipac.caltech.edu/2mass/)

Special thanks to the astronomical community for making these datasets publicly available.
