# Docker Usage Guide

Complete guide for using Docker containers with the Gaia EB Teff pipeline.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Images](#docker-images)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Docker Compose](#docker-compose)
- [Volume Mounts](#volume-mounts)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 2. Run prediction (lightweight)
docker compose run --rm predict --help

# 3. Train a model (requires data in data/raw or data/processed)
docker compose run --rm train --ml --ml-config config/models/gaia_teff_corrected_log_optuna.yaml

# 4. Generate validation plots
docker compose run --rm validate --validate --val-config config/validation/validate_gaia_teff_flag1_corrected_optuna.yaml
```

## Docker Images

Two specialized images for different workloads:

### Prediction Image (`Dockerfile`)

**Size**: ~943MB
**Purpose**: Lightweight inference container
**Use Cases**:
- Making predictions on new data
- Production deployments
- Quick testing

**Includes**:
- Python 3.11
- Core ML libraries (scikit-learn, pandas, polars)
- Data formats (parquet, FITS)
- HuggingFace Hub integration
- **Excludes**: matplotlib, seaborn (for smaller size)

**Build**:
```bash
docker build -f Dockerfile -t gaia-eb-teff:predict .
```

### Training Image (`Dockerfile.train`)

**Size**: ~1.84GB
**Purpose**: Full training and validation environment
**Use Cases**:
- Training new models
- Generating validation plots
- Full pipeline execution

**Includes**:
- Everything in prediction image
- Visualization libraries (matplotlib, seaborn)
- Jupyter (optional for interactive work)
- Full astronomy tools

**Build**:
```bash
docker build -f Dockerfile.train -t gaia-eb-teff:train .
```

## Building Images

### Build Both Images

```bash
# Prediction image
docker build -f Dockerfile -t gaia-eb-teff:predict .

# Training image
docker build -f Dockerfile.train -t gaia-eb-teff:train .
```

### Build with Docker Compose

```bash
# Build all services
docker compose build

# Build specific service
docker compose build train
docker compose build predict
```

### Build Options

```bash
# No cache (force rebuild)
docker build --no-cache -f Dockerfile -t gaia-eb-teff:predict .

# Specify platform (for Apple Silicon)
docker build --platform linux/amd64 -f Dockerfile -t gaia-eb-teff:predict .

# Build with build args (if needed)
docker build --build-arg PYTHON_VERSION=3.11 -f Dockerfile -t gaia-eb-teff:predict .
```

## Running Containers

### Basic Usage

```bash
# Show help
docker run --rm gaia-eb-teff:predict --help

# Run with specific command
docker run --rm gaia-eb-teff:predict --ml --dry-run
```

### With Volume Mounts

```bash
# Mount data directory
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  gaia-eb-teff:predict \
  --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Mount all directories
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/config:/app/config \
  gaia-eb-teff:train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml
```

### With Environment Variables

```bash
# Set HF_TOKEN for auto-download
docker run --rm \
  -e HF_TOKEN=your_token_here \
  -v $(pwd)/data:/app/data \
  gaia-eb-teff:predict \
  --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Use .env file
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  gaia-eb-teff:predict \
  --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml
```

### Interactive Shell

```bash
# Start interactive bash session
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  gaia-eb-teff:train \
  /bin/bash

# Then inside container:
python pipeline.py --help
ls data/
```

## Docker Compose

Docker Compose simplifies multi-container workflows.

### Available Services

- **`train`**: Full training environment
- **`predict`**: Lightweight prediction service
- **`validate`**: Validation plot generation
- **`data`**: Data processing service

### Common Commands

```bash
# Show services
docker compose config --services

# Run a service once and remove
docker compose run --rm train --help
docker compose run --rm predict --help

# Start service in background
docker compose up -d train

# View logs
docker compose logs train
docker compose logs -f predict  # Follow logs

# Stop services
docker compose stop
docker compose down  # Stop and remove containers
```

### Training Workflow

```bash
# Set environment variables
export HF_TOKEN=your_token

# Train a model
docker compose run --rm train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Or use default command in docker-compose.yml
docker compose up train
```

### Prediction Workflow

```bash
# Run predictions with default config
docker compose run --rm predict \
  --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# With custom model
MODEL_NAME=rf_gaia_2mass_ir_20251103_141119 \
  docker compose run --rm predict \
  --predict --pred-config config/prediction/predict_gaia_2mass_ir.yaml
```

### Validation Workflow

```bash
# Generate validation plots
docker compose run --rm validate \
  --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml

# Output will be in reports/figures/
ls reports/figures/
```

## Volume Mounts

### Recommended Volume Structure

```
project/
├── data/          # Mount to /app/data (read/write for training)
├── models/        # Mount to /app/models (read-only for prediction)
├── reports/       # Mount to /app/reports (write for validation)
└── predictions/   # Mount to /app/predictions (write for new predictions)
```

### Volume Mount Options

```bash
# Read-only mount (for production prediction)
-v $(pwd)/models:/app/models:ro

# Read-write mount (for training)
-v $(pwd)/data:/app/data

# Named volumes (persistent across containers)
docker volume create gaia_data
docker run -v gaia_data:/app/data gaia-eb-teff:train
```

### Docker Compose Volumes

Defined in `docker-compose.yml`:

```yaml
volumes:
  - ./data:/app/data           # Data directory
  - ./models:/app/models       # Model files
  - ./reports:/app/reports     # Validation reports
  - ./config:/app/config       # Configuration files
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | `hf_xxxxx...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Model to auto-download | None |
| `DATA_CACHE_DIR` | Data cache directory | `/app/data/processed` |
| `MODEL_CACHE_DIR` | Model cache directory | `/app/models` |
| `KAGGLE_USERNAME` | Kaggle username | None |
| `KAGGLE_KEY` | Kaggle API key | None |

### Setting Environment Variables

**Option 1: .env file** (recommended)
```bash
cp .env.example .env
# Edit .env with your credentials
docker compose run --rm train --ml --ml-config config/models/gaia_teff_corrected_log.yaml
```

**Option 2: Command line**
```bash
docker run --rm \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=rf_gaia_teff_corrected_log_20251126_130144 \
  gaia-eb-teff:predict --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml
```

**Option 3: --env-file**
```bash
docker run --rm --env-file .env gaia-eb-teff:predict --help
```

## Advanced Usage

### Multi-Stage Pipelines

```bash
# 1. Download data
docker compose run --rm data --data

# 2. Train model
docker compose run --rm train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# 3. Make predictions
docker compose run --rm predict \
  --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# 4. Validate
docker compose run --rm validate \
  --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml
```

### Resource Limits

```bash
# Limit memory
docker run --rm \
  --memory=8g \
  --memory-swap=16g \
  -v $(pwd)/data:/app/data \
  gaia-eb-teff:train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Limit CPUs
docker run --rm \
  --cpus=4 \
  -v $(pwd)/data:/app/data \
  gaia-eb-teff:train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml
```

### Background Execution

```bash
# Start in background
docker compose up -d train

# Monitor logs
docker compose logs -f train

# Check status
docker compose ps

# Stop when done
docker compose stop train
```

## Troubleshooting

### Issue: Out of Memory

**Symptoms**: Container crashes, "Killed" messages

**Solutions**:
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → Increase to 8GB+

# Or use --memory flag
docker run --memory=8g --memory-swap=16g ...

# Or reduce model size in config
# Edit config/models/*.yaml: reduce n_estimators, max_depth
```

### Issue: Slow Dataset Downloads

**Symptoms**: Long wait times, timeouts

**Solutions**:
```bash
# Pre-download datasets before running container
python scripts/download_datasets.py --datasets training

# Or increase timeout (if needed)
# Edit docker-entrypoint.sh to add timeout handling
```

### Issue: Permission Errors

**Symptoms**: "Permission denied" when writing to volumes

**Solutions**:
```bash
# Run as current user (Linux/Mac)
docker run --user $(id -u):$(id -g) ...

# Or fix permissions on host
chmod -R 755 data/ models/ reports/
```

### Issue: Model Not Found

**Symptoms**: "Model file not found" errors

**Solutions**:
```bash
# Download model manually
python scripts/download_datasets.py --model gaia_teff_corrected_log

# Or set MODEL_NAME environment variable
export MODEL_NAME=rf_gaia_teff_corrected_log_20251126_130144
docker compose run --rm predict ...

# Check model registry
cat config/models/model_registry.yaml
```

### Issue: HuggingFace Authentication

**Symptoms**: 401 errors, "Authentication required"

**Solutions**:
```bash
# Login via CLI (on host)
huggingface-cli login

# Or set HF_TOKEN
export HF_TOKEN=your_token
docker compose run --rm predict ...

# Or add to .env file
echo "HF_TOKEN=your_token" >> .env
```

### Issue: Config File Not Found

**Symptoms**: "Configuration file not found"

**Solutions**:
```bash
# Ensure config directory is mounted
-v $(pwd)/config:/app/config

# Check file exists on host
ls config/models/
ls config/prediction/
ls config/validation/

# Or use absolute path
docker run -v /full/path/to/config:/app/config ...
```

### Issue: Validation Plots Fail in Prediction Container

**Symptoms**: "ModuleNotFoundError: No module named 'matplotlib'"

**Solution**: Use training container for validation:
```bash
# Wrong (prediction container)
docker compose run --rm predict --validate ...

# Correct (training container)
docker compose run --rm validate --validate ...
```

### Debug Mode

```bash
# Enable verbose logging
docker compose run --rm train --verbose --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Check container filesystem
docker run -it --rm gaia-eb-teff:train /bin/bash
ls -la /app/
cat /app/pipeline.py
```

## Production Deployment

### Docker Registry

```bash
# Tag for registry
docker tag gaia-eb-teff:predict your-registry.com/gaia-eb-teff:predict
docker tag gaia-eb-teff:train your-registry.com/gaia-eb-teff:train

# Push to registry
docker push your-registry.com/gaia-eb-teff:predict
docker push your-registry.com/gaia-eb-teff:train
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests (if available).

### CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
name: Build Docker Images
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build images
        run: |
          docker build -f Dockerfile -t gaia-eb-teff:predict .
          docker build -f Dockerfile.train -t gaia-eb-teff:train .
```

## Best Practices

1. **Use .env for secrets**: Never commit credentials to git
2. **Read-only mounts for production**: Use `:ro` for model/config volumes
3. **Resource limits**: Always set memory/CPU limits in production
4. **Health checks**: Monitor container health with `docker ps`
5. **Log aggregation**: Use `docker logs` or external logging service
6. **Version tagging**: Tag images with version numbers or git commits
7. **Minimal images for production**: Use prediction image for inference
8. **Regular updates**: Rebuild images when dependencies update

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/YOUR_ORG/gaia-eb-teff/issues
- Documentation: See README.md and DATASET_ACCESS.md
