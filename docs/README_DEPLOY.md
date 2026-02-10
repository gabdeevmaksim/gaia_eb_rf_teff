# Gaia Eclipsing Binary Temperature Prediction Pipeline

Production-ready ML pipeline for predicting effective temperatures of eclipsing binary stars using Gaia, Pan-STARRS, SDSS, and 2MASS photometry.

**Deployment Branch** - This is the production-ready branch with Docker containerization and automated dataset downloads. For development work, see the `main` branch.

## Features

- **5 Complete Pipelines**: Data processing, ML training, prediction, validation, complete workflow
- **30+ Pre-configured Models**: YAML-driven model configurations for different photometric combinations
- **Best-of-Three Ensemble**: 263K mean uncertainty (18% improvement vs single model)
- **Docker Support**: Training and prediction containers with auto-downloads
- **HuggingFace Integration**: Automatic dataset and model downloads from HuggingFace Hub
- **Production Ready**: Logging, error handling, versioned models, reproducible results

## Dataset

**Full Catalog**: 2.1M eclipsing binaries with Teff predictions (97.2% coverage)
- HuggingFace: [YOUR_ORG/gaia-eb-teff-datasets](https://huggingface.co/datasets/Dedulek/gaia-eb-teff-datasets)
- 196MB FITS file with quality flags (A=Gaia, B=ML<300K, C=ML<500K, D=ML≥500K)

**Pre-trained Models** (1.2-2GB each):
- **gaia_teff_corrected_log** (RECOMMENDED): 556K MAE, R²=0.640
- **gaia_2mass_ir**: Optical + infrared, 765K MAE
- **gaia_all_colors_teff_log**: Log-transformed, 557K MAE

See [DATASET_ACCESS.md](DATASET_ACCESS.md) for download instructions.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository (deploy branch)
git clone https://github.com/gabdeevmaksim/gaia-eb-teff.git -b deploy
cd gaia-eb-teff

# Setup environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Train a model
docker-compose up train

# Run predictions
docker-compose up predict

# Generate validation plots
docker-compose up validate
```

See [DOCKER_USAGE.md](DOCKER_USAGE.md) for detailed Docker instructions.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/gaia-eb-teff.git -b deploy
cd gaia-eb-teff

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python scripts/download_datasets.py --datasets training

# Train a model
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Run predictions
python pipeline.py --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Validate model
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml
```

## Pipeline Usage

### 1. Training Pipeline

Train a new model using configuration files:

```bash
# Using Docker
docker-compose run train --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Local
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log.yaml
```

**Output**: Model saved to `models/` with timestamped filename, metadata, summary, and test predictions.

### 2. Prediction Pipeline

Generate predictions for new data:

```bash
# Using Docker (auto-downloads model)
docker-compose run predict --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Local
python pipeline.py --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml
```

**Output**: Predictions saved to `data/processed/` as parquet file.

### 3. Validation Pipeline

Generate validation plots and metrics:

```bash
# Using Docker
docker-compose run validate --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml

# Local
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml
```

**Output**: Plots saved to `reports/figures/`, metrics saved to `reports/`.

### 4. Complete Pipeline (Train → Validate → Predict)

Run all three steps in one command using bash command chaining:

**Using Docker** (recommended):

```bash
# Single command for complete workflow
docker compose run --rm train --ml --ml-config config/models/YOUR_CONFIG.yaml && \
docker compose run --rm train --validate --val-config config/validation/YOUR_VALIDATION.yaml && \
docker compose run --rm train --predict --pred-config config/prediction/YOUR_PREDICTION.yaml
```

**Local Installation**:

```bash
# Activate environment first
source .venv/bin/activate

# Run complete pipeline
python pipeline.py --ml --ml-config config/models/YOUR_CONFIG.yaml && \
python pipeline.py --validate --val-config config/validation/YOUR_VALIDATION.yaml && \
python pipeline.py --predict --pred-config config/prediction/YOUR_PREDICTION.yaml
```

**Example: Multi-Survey Feature Engineering Pipeline**

```bash
# Train with Gaia + Pan-STARRS + 2MASS features
docker compose run --rm train --ml --ml-config config/models/test_multisurvey_features.yaml && \
docker compose run --rm train --validate --val-config config/validation/validate_multisurvey_test.yaml && \
docker compose run --rm train --predict --pred-config config/prediction/predict_multisurvey_test.yaml
```

**Expected Outputs**:
- **Training**: Model saved to `models/rf_*_TIMESTAMP.pkl` + metadata + summary
- **Validation**: 5 plots in `reports/figures/YOUR_MODEL_validation/` + text report
- **Predictions**: Parquet file in `data/processed/predictions_*.parquet`

**Typical Timing** (for 1.75M training samples, 100 trees):
- Training: 5-10 minutes (includes feature engineering)
- Validation: 10-30 seconds
- Prediction: 2-5 minutes (depends on dataset size and filtering)

**Note**: The `&&` operator ensures each step only runs if the previous step succeeds. If training fails, validation and prediction will not run.

### 5. Built-in Complete Pipeline

Alternatively, run the full data processing + training + validation + prediction workflow:

```bash
python pipeline.py --all
```

This runs all built-in pipelines with default configurations. For custom workflows, use the chained commands above.

## Configuration

All pipelines are configured via YAML files in `config/`:

- **`config/models/*.yaml`**: Model training configurations (30 files)
  - Define features, hyperparameters, target transformations
  - Example: `gaia_teff_corrected_log.yaml`

- **`config/prediction/*.yaml`**: Prediction configurations (18 files)
  - Specify model, input data, preprocessing
  - Supports wildcard model matching
  - **Template**: `template_prediction.yaml`

- **`config/validation/*.yaml`**: Validation configurations (20 files)
  - Define plots, metrics, output locations
  - **Template**: `template_validation.yaml`

- **`config/config.yaml`**: Central configuration
  - Paths, datasets, default parameters

See [docs/CONFIGURABLE_PIPELINE.md](docs/CONFIGURABLE_PIPELINE.md) for configuration details.

### Creating Custom Configurations from Templates

**Step 1: Copy the template files**

```bash
# Copy prediction template
cp config/prediction/template_prediction.yaml config/prediction/my_prediction.yaml

# Copy validation template
cp config/validation/template_validation.yaml config/validation/my_validation.yaml
```

**Step 2: Customize for your model**

Edit `config/prediction/my_prediction.yaml`:
- Set `model_file` to your trained model (supports wildcards: `rf_my_model_*.pkl`)
- Update `required_features` to match your training features
- Set `feature_engineering.enabled` to match training configuration
- Optionally add `filters` (e.g., `{teff_gaia: -999.0}` to only predict for objects without Teff)
- Customize `output_file` and `include_columns`

Edit `config/validation/my_validation.yaml`:
- Set `model_pattern` to match your model files (e.g., `rf_my_model_*`)
- Customize `figures_subdir` for output location
- Set `report_file` path for text metrics
- Adjust `target_info` (name, unit, short) if predicting something other than Teff

**Step 3: Run the complete pipeline**

```bash
# Train your model first
python pipeline.py --ml --ml-config config/models/my_model.yaml

# Then run validation and prediction
python pipeline.py --validate --val-config config/validation/my_validation.yaml && \
python pipeline.py --predict --pred-config config/prediction/my_prediction.yaml
```

**Important Notes**:
- **Feature Engineering**: Prediction config MUST match training config exactly!
  - If training used `feature_engineering.enabled: true`, prediction must too
  - All settings (`color_cols`, `mag_cols`, `polynomial_degree`, `interactions`) must match
- **Target Transform**: If training used log transform, prediction automatically inverse-transforms
- **Filters**: Production predictions should filter to objects without existing values (saves compute)
- **Wildcards**: Model matching uses glob patterns - `rf_my_model_*` finds most recent timestamped version

## Model Performance

| Model | MAE (K) | RMSE (K) | R² | Within 10% |
|-------|---------|----------|-----|------------|
| Gaia Teff Corrected Log (BEST) | 557 | 1021 | 0.640 | 68.5% |
| Gaia + 2MASS IR | 765 | 1168 | 0.315 | 43.4% |
| Gaia All Colors Log | 557 | 1021 | 0.640 | 68.5% |
| Ensemble PanSTARRS | 720 | 1184 | 0.297 | 53.0% |

## Directory Structure

```
deploy/
├── pipeline.py              # Master orchestrator
├── requirements.txt         # Full dependencies
├── requirements-docker.txt  # Minimal for containers
├── Dockerfile               # Prediction container
├── Dockerfile.train         # Training container
├── docker-compose.yml       # Multi-service orchestration
├── docker-entrypoint.sh     # Container startup script
├── .env.example             # Environment template
│
├── src/                     # Source modules
│   ├── config/              # Settings management
│   ├── data/                # Data loading, caching
│   ├── features/            # Feature engineering
│   ├── visualization/       # Plotting utilities
│   └── pipeline/            # Pipeline implementations
│
├── config/                  # Configuration
│   ├── config.yaml          # Central config
│   ├── models/              # Training configs (30)
│   ├── prediction/          # Prediction configs (18)
│   └── validation/          # Validation configs (20)
│
├── docs/                    # Documentation
│   ├── PIPELINES.md
│   ├── CONFIGURABLE_PIPELINE.md
│   ├── CONFIGURATION.md
│   └── ...
│
└── scripts/                 # Essential scripts
    ├── download_datasets.py
    └── upload_to_huggingface.py
```

## Documentation

- **[DOCKER_USAGE.md](DOCKER_USAGE.md)**: Complete Docker guide (building, running, troubleshooting)
- **[DATASET_ACCESS.md](DATASET_ACCESS.md)**: Dataset download methods and structure
- **[docs/PIPELINES.md](docs/PIPELINES.md)**: Pipeline architecture and usage
- **[docs/CONFIGURABLE_PIPELINE.md](docs/CONFIGURABLE_PIPELINE.md)**: Model configuration guide
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Configuration system API

## Requirements

**Python**: 3.9+

**Core Dependencies**:
- numpy, pandas, scikit-learn, joblib
- polars, pyarrow (data formats)
- astropy (FITS files)
- pyyaml (configuration)
- huggingface_hub (dataset downloads)

**Optional** (for validation plots):
- matplotlib, seaborn

**Docker**: For containerized deployment (recommended)

## Environment Variables

Create `.env` file from `.env.example`:

```bash
# Required for dataset/model downloads
HF_TOKEN=your_huggingface_token

# Optional for specific model
MODEL_NAME=rf_gaia_teff_corrected_log_20251126_130144

# Optional: Kaggle credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Examples

### Example 1: Train Custom Model

```bash
# Create custom config: config/models/my_custom_model.yaml
# Then train:
python pipeline.py --ml --ml-config config/models/my_custom_model.yaml
```

### Example 2: Batch Predictions

```bash
# Prepare your input data as parquet file
# Create prediction config
# Run predictions
python pipeline.py --predict --pred-config config/prediction/my_predictions.yaml
```

### Example 3: Model Comparison

```bash
# Train multiple models
for config in config/models/*.yaml; do
    python pipeline.py --ml --ml-config $config
done

# Validate all
for config in config/validation/*.yaml; do
    python pipeline.py --validate --val-config $config
done
```

## Troubleshooting

### Issue: HuggingFace download fails

**Solution**: Ensure `HF_TOKEN` is set and valid:
```bash
export HF_TOKEN=your_token
# Or login via CLI:
huggingface-cli login
```

### Issue: Out of memory during training

**Solution**: Reduce `n_estimators` or `max_depth` in model config, or use Docker with memory limits:
```bash
docker run --memory=8g --memory-swap=16g ...
```

### Issue: Model not found

**Solution**: Check model registry and download:
```bash
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

## Citation

If you use this pipeline or dataset, please cite:

```bibtex
@article{your_paper,
  title={Effective Temperature Predictions for Eclipsing Binary Stars},
  author={Your Name},
  journal={Journal},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

This is the deployment branch. For development contributions, please work on the `main` branch and submit pull requests there.

## Support

- **Issues**: https://github.com/YOUR_ORG/gaia-eb-teff/issues
- **Discussions**: https://github.com/YOUR_ORG/gaia-eb-teff/discussions
- **Email**: your.email@example.com

## Acknowledgments

- **Gaia Mission**: ESA's Gaia satellite (DR3)
- **Pan-STARRS**: Panoramic Survey Telescope and Rapid Response System
- **SDSS**: Sloan Digital Sky Survey
- **2MASS**: Two Micron All Sky Survey
