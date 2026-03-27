# How to run from scratch (Teff EB-params Optuna model)

This is a minimal end-to-end setup: create a venv, install deps, download the dataset + Teff correction coefficients from Hugging Face, train the model using `config/models/gaia_teff_eb_params_optuna.yaml`, validate the trained model, and generate predictions.

## 1) Create venv + install requirements

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Download datasets + Teff correction coefficients

Download the unified photometry input (to `data/raw/`), predictions dataset (to `data/processed/`), and Teff correction coefficients (to `data/`):

```bash
python scripts/download_datasets.py --datasets all
```

After this, you should have at least:

```bash
ls -lh data/raw/eb_unified_photometry.parquet
ls -lh data/teff_correction_coeffs_deg2.pkl
```

## 3) Train the model

Train using the provided model config:

```bash
python pipeline.py --ml --ml-config config/models/gaia_teff_eb_params_optuna.yaml
```

This will create a new model in `models/` with an ID like:

```text
rf_gaia_teff_eb_params_optuna_YYYYMMDD_HHMMSS.pkl
```

## 4) Validate the newly trained model

Validation config (already prepared):
- `config/validation/validate_gaia_teff_eb_params_optuna.yaml`

Run validation:

```bash
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_eb_params_optuna.yaml
```

Outputs:
- Figures: `reports/figures/gaia_teff_eb_params_optuna_validation/`
- Report: `reports/validation_report_gaia_teff_eb_params_optuna.txt`


