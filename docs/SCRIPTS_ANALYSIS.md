# Scripts Directory Analysis

Overview of all scripts in `scripts/`: purpose, config usage, inputs/outputs, and recommendations.

---

## Summary Table

| Script | Purpose | Config usage | Inputs | Outputs |
|--------|---------|--------------|--------|--------|
| `download_datasets.py` | Download from HuggingFace | None (PROJECT_ROOT) | HF Hub | `data/raw/`, `data/`, `data/processed/`, `models/` |
| `upload_to_huggingface.py` | Upload to HuggingFace | None | Local paths | HF Hub |
| `upload_readmes.py` | Upload HF READMEs | None | `docs/HF_*.md` | HF Hub |
| `merge_teff_predictions_into_unified.py` | Merge 4 prediction parquets + photometry | Full | raw photometry, processed preds | `eb_catalog_teff.parquet` |
| `profile_dataset.py` | ydata_profiling HTML reports | Datasets + processing | Config key or file path | `reports/profiles/*.html` |
| `evaluate_flag1_on_shared_test.py` | Flag1 metrics on shared test set | Paths + model config | Photometry, flag1 predictions | Metrics file |
| `evaluate_chained_model.py` | Chain model metrics vs GSP-Phot | Paths + model config | Photometry, chain predictions | Metrics + optional file |
| `plot_teff_best_uncertainty_histogram.py` | Histogram of best Teff uncertainty | Paths | `eb_catalog_teff.parquet` | `paper/figures/` |
| `summarize_model_test_metrics.py` | MAE/RMSE/R² table for 4 models | Paths | Model metadata + chain file + catalog | `paper/model_comparison_test_metrics.md` |
| `create_paper_temperature_histograms.py` | Paper histograms (Gaia vs ML, uncertainties) | Paths | `eb_catalog_teff.parquet` | `paper/figures/` |
| `propagate_uncertainty.py` | Chained logg→Teff with MC uncertainty | Paths | Photometry, logg + Teff models | `teff_propagated_uncertainty.parquet` |
| `create_paper_validation_plots.py` | 4 validation plots (importance, true vs pred, etc.) | Paths (models, project_root) | Model metadata + test predictions | `paper/figures/` |

---

## 1. HuggingFace (download / upload)

### `download_datasets.py`
- **Purpose:** Download datasets and/or models from HuggingFace Hub.
- **Implementation:** Thin CLI; logic in `src.huggingface.download`.
- **Paths:** `--datasets training` → `data/raw/`; `correction` → `data/`; `catalog` / `all` → `--output-data` (default `data/processed`). Models → `--output-models` (default `models`).
- **Config:** None; uses `PROJECT_ROOT` only.
- **Run from:** Project root.

### `upload_to_huggingface.py`
- **Purpose:** Upload datasets (photometry, predictions, catalog) and/or models; optional `--clean`, `--models-from-dir`.
- **Implementation:** `src.huggingface` (clean_repo, upload_datasets, upload_models).
- **Config:** None.
- **Run from:** Project root.

### `upload_readmes.py`
- **Purpose:** Upload `docs/HF_DATASET_README.md` and `docs/HF_MODEL_README.md` to the Hub.
- **Implementation:** `src.huggingface.upload_readmes`.
- **Config:** None.

---

## 2. Data merge and profiling

### `merge_teff_predictions_into_unified.py`
- **Purpose:** Build the final catalog with Teff: load `eb_unified_photometry` (raw), apply GSP-Phot Teff correction, left-join four prediction parquets, compute `teff_best` and `teff_best_uncertainty`.
- **Config:** `get_dataset_path("eb_unified_photometry", "raw")`, `get_dataset_path("eb_catalog_teff", "processed")`, `get_path("processed")`, `get("processing", "missing_value")`. Correction coeffs: `data_root / teff_correction_coeffs_deg2.pkl`.
- **Hardcoded:** `PREDICTION_FILES` (filenames and column mappings). Any new model or rename requires editing this list.
- **Inputs:** Unified photometry (default from config raw), processed dir (default config processed), correction pkl in `data/`.
- **Output:** `data/processed/eb_catalog_teff.parquet` (default).

### `profile_dataset.py`
- **Purpose:** Generate ydata_profiling HTML reports for exploratory analysis.
- **Config:** `get_dataset_path(dataset_key, location)`, `get('processing', 'missing_value')`. With `--all`, iterates over all keys in `config.datasets`.
- **Fixed:** Docstring and `--dataset` help now use current config keys (`eb_unified_photometry`, `eb_catalog_teff`) and document `--location raw` for photometry.

---

## 3. Evaluation (test-set metrics)

### `evaluate_flag1_on_shared_test.py`
- **Purpose:** Evaluate the Flag1 model on the same test set as clustering/log/chain (reproduce split from a reference model config).
- **Config:** `get_dataset_path("eb_unified_photometry", "raw")`, `get_path("processed")`. Reference model config default: `config/models/gaia_teff_corrected_log_optuna.yaml`.
- **Requires:** Predictions from `predict_gaia_original_teff_flag1_corrected_optuna.yaml` (objects with original Teff). Imports `evaluate_chained_model` for `get_teff_model_test_set` and `load_model_config`.

### `evaluate_chained_model.py`
- **Purpose:** Evaluate chained (logg → Teff) model on the same test set as the Gaia+logg Teff model; compare to GSP-Phot.
- **Config:** `get_dataset_path("eb_unified_photometry", "raw")`, `get_path("data_root")` for correction coeffs. Default model config: `config/models/gaia_logg_teff_corrected_log_optuna.yaml`. Default predictions: `data/processed/teff_propagated_uncertainty.parquet`.
- **Logic:** Reuses train/test split and preprocessing from model config (filters, Teff correction, drop_missing, test_size, random_state).

---

## 4. Paper figures and summaries

### `plot_teff_best_uncertainty_histogram.py`
- **Purpose:** Histogram of `teff_best_uncertainty` from the final catalog.
- **Config:** `get_dataset_path("eb_catalog_teff", "processed")`, `get_path("processed")`. Output dir: `--output-dir` or `PROJECT_ROOT / "paper" / "figures"`.

### `create_paper_temperature_histograms.py`
- **Purpose:** Publication histograms: Gaia-only vs ML (best) Teff, and ML uncertainty distribution.
- **Config:** `get_dataset_path("eb_catalog_teff", "processed")`. Output: `paper/figures/` (not from config).

### `summarize_model_test_metrics.py`
- **Purpose:** Build markdown/CSV table of MAE, RMSE, R² for the four Teff models (clustering, log, flag1, chain) and optional “% chosen as best” from catalog.
- **Config:** `get_path("models")`, and catalog path for “pct chosen” (from config processed + `eb_catalog_teff`).
- **Hardcoded:** `MODEL_PATTERNS`, `CHAIN_METRICS_FILE`, `PREDICTION_PREFIXES` — must match merge script and actual model stems.

### `create_paper_validation_plots.py`
- **Purpose:** Four validation plots: feature importance, true vs predicted, RMSE by Teff range, % within 10%.
- **Config:** **Fixed.** Uses `config.get_path("models")` for `MODEL_DIR` and `config.project_root / "paper" / "figures"` for default output dir; `sys.path` and `get_config()` added so script is runnable from any cwd.

---

## 5. Chained prediction with uncertainty

### `propagate_uncertainty.py`
- **Purpose:** Chained prediction (logg → Teff) with Monte Carlo uncertainty propagation; writes mean and std Teff per object.
- **Config:** `get_config()`, `get_path("models")`, `get_path("processed")`, `get_dataset_path("eb_unified_photometry", "raw")` for default data path.
- **Models:** By default picks latest `rf_gaia_logg_optuna_*.pkl` and `rf_gaia_logg_teff_corrected_log_optuna_*.pkl` (excludes clustering/scaler).
- **Output:** `data/processed/teff_propagated_uncertainty.parquet` (default).

---

## Recommendations

1. ~~**profile_dataset.py:** Update docstring and help~~ **Done:** docstring and `--dataset` help now use `eb_unified_photometry` / `eb_catalog_teff` and `--location raw`.
2. ~~**create_paper_validation_plots.py:** Use config for paths~~ **Done:** uses `config.get_path("models")` and `config.project_root / "paper" / "figures"`.
3. **Merge script:** Document that adding or renaming a model requires updating `PREDICTION_FILES` in `merge_teff_predictions_into_unified.py` and the corresponding prefixes in `summarize_model_test_metrics.py`.
4. **Naming:** Several scripts write under `paper/figures/` while config has `reports/figures`. Consider documenting the distinction (e.g. “paper = manuscript figures, reports/figures = pipeline/validation”) or unifying in config.
5. **download_datasets.py:** Default `--output-data` is `data/processed`; for `--datasets training` the script overrides and uses `data/raw`. Docstring is correct; no change required.

---

## Dependency Overview

- **Pipeline vs scripts:** Training, prediction, and validation are run via `pipeline.py` and configs under `config/models`, `config/prediction`, `config/validation`. Scripts do not drive the main ML pipeline; they do merge, evaluation, profiling, and paper figures.
- **Order for full catalog:**  
  1) Download photometry (and optionally correction) → `data/raw`.  
  2) Run prediction pipeline for each model; run `propagate_uncertainty.py` for chain.  
  3) `merge_teff_predictions_into_unified.py` → `eb_catalog_teff.parquet`.  
  4) Optional: `summarize_model_test_metrics.py`, `plot_teff_best_uncertainty_histogram.py`, `create_paper_temperature_histograms.py`, `create_paper_validation_plots.py`.
- **Shared code:** `evaluate_flag1_on_shared_test.py` imports test-set and config-loading helpers from `evaluate_chained_model.py`.
