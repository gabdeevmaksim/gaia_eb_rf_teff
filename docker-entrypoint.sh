#!/bin/bash
set -e

echo "Starting Gaia EB Teff Pipeline..."

# Auto-download datasets if not present and HF_TOKEN is set
if [ ! -z "$HF_TOKEN" ]; then
    if [ ! -f "data/processed/eb_unified_photometry.parquet" ]; then
        echo "Training data not found. Downloading from HuggingFace..."
        python scripts/download_datasets.py --datasets training || echo "Warning: Could not download datasets"
    else
        echo "Training data found. Skipping download."
    fi

    # Auto-download Teff correction coefficients if not present (file lives under data/)
    if [ ! -f "data/teff_correction_coeffs_deg2.pkl" ]; then
        echo "Teff correction coefficients not found. Downloading from HuggingFace..."
        python scripts/download_datasets.py --datasets correction || echo "Warning: Could not download correction coefficients"
    else
        echo "Teff correction coefficients found. Skipping download."
    fi
else
    echo "HF_TOKEN not set. Skipping auto-download. Set HF_TOKEN to enable automatic dataset downloads."
fi

# Auto-download model if MODEL_NAME is specified
if [ ! -z "$MODEL_NAME" ]; then
    if [ ! -f "models/${MODEL_NAME}.pkl" ]; then
        echo "Model not found: $MODEL_NAME"
        echo "Downloading from HuggingFace..."
        python scripts/download_datasets.py --model "$MODEL_NAME" || echo "Warning: Could not download model"
    else
        echo "Model found: $MODEL_NAME"
    fi
fi

# Execute the main command (pipeline.py with provided arguments)
echo "Executing: python pipeline.py $@"
exec python pipeline.py "$@"
