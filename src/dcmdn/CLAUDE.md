# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Convolutional Mixture Density Network (DCMDN) - a research implementation for probabilistic regression using Gaussian mixture outputs. Primary application: astronomical redshift estimation from SDSS quasar images.

**Framework:** Theano (legacy, discontinued)
**License:** CC BY-NC-SA 4.0 (non-commercial use only)
**Authors:** Kai Polsterer & Antonio D'Isanto (HITS gGmbH, 2017)

## Running the Code

```bash
# Activate environment first (Theano required)
python Experiments.py
```

**Required data files (not in repo):**
- `SDSS_quasar_catalog_all_coord.csv` - catalog with redshift labels
- `compressedImageCatalog.npz` - numpy compressed image archive

## Architecture

```
Input (15ch × 16×16) → 4 ConvLayers (128→256→512→1024) → MDN Head → Gaussian Mixture (5 components)
```

**Core modules:**
- `DeepConvolutionalMixtureDensityNetwork.py` - Main orchestrator, training loop with 60% dropout
- `ConvolutionalLayer.py` / `ConvolutionalPoolingLayer.py` - Feature extraction layers
- `MixtureDensityNetwork.py` - Dense regression head
- `LossFunctions.py` - CRPS and negative log-likelihood losses, PIT for calibration
- `ImageAccess.py` - Data loading, generates 15 channels from 5 bands via band differences
- `WeightInitialization.py` - Orthogonal and LeCun initialization

**Key hyperparameters (hardcoded in Experiments.py):**
- Batch size: 1000
- Learning rate: 0.01
- Epochs: 1000
- Gaussian components: 5
- Validation interval: 5 epochs

## ML Pipeline

```
Load Data → Build Colors (5 bands → 15 channels) → Split (100k/20k/65k) →
Train with Dropout → Validate → Checkpoint (.npy) → Predict → Analyze (CRPS, PIT)
```

**Loss functions:**
- `GaussianMixtureCRPS` - Continuous Ranked Probability Score (robust)
- `negativeLogLikelyhood` - Standard likelihood (faster)

**Model evaluation:**
- CRPS metric for prediction quality
- PIT histogram for calibration assessment

## Important Patterns

- All layers implement: `getOutputs()`, `getParameters()`, `getWeights()`, `getBiases()`
- Uses Theano shared variables for GPU compatibility
- RandomState passed for reproducibility (seeds: 23456, 35325)
- Model checkpoints: `{prefix}_{epoch}_{train_loss}_{val_loss}.npy`

## Modernization Notes

This is Python 2.7 era code using Theano. For active development:
- Migrate to PyTorch or TensorFlow
- Create config.yaml for hyperparameters (per project conventions)
- Add requirements.txt
