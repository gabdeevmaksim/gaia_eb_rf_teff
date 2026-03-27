#!/usr/bin/env python3
"""
Download datasets and models from HuggingFace Hub.

Implementation in src.huggingface. Run from project root.

Usage:
    python scripts/download_datasets.py --datasets training
    python scripts/download_datasets.py --datasets catalog --model all
    python scripts/download_datasets.py --datasets all --model all
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from src.huggingface import download_from_huggingface, download_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        choices=["photometry", "predictions", "training", "correction", "all"],
        help="Dataset: photometry (raw), predictions (eb_catalog_teff), training (=photometry), correction, all",
    )
    parser.add_argument(
        "--model",
        help="Model name to download (or 'all')",
    )
    parser.add_argument(
        "--output-data",
        default="data/processed",
        help="Output directory for datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output-models",
        default="models",
        help="Output directory for models (default: models)",
    )

    args = parser.parse_args()

    if not args.datasets and not args.model:
        parser.print_help()
        print("\nError: Specify --datasets and/or --model")
        sys.exit(1)

    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. Public datasets work; private need token.")

    success = True

    if args.datasets:
        if args.datasets == "all":
            # photometry → raw, predictions → processed, correction → data/
            if not download_from_huggingface("photometry", PROJECT_ROOT / "data" / "raw", PROJECT_ROOT):
                success = False
            if not download_from_huggingface("predictions", PROJECT_ROOT / "data" / "processed", PROJECT_ROOT):
                success = False
            if not download_from_huggingface("correction", PROJECT_ROOT / "data", PROJECT_ROOT):
                success = False
        elif args.datasets == "correction":
            if not download_from_huggingface("correction", PROJECT_ROOT / "data", PROJECT_ROOT):
                success = False
        elif args.datasets in ("photometry", "training"):
            if not download_from_huggingface(args.datasets, PROJECT_ROOT / "data" / "raw", PROJECT_ROOT):
                success = False
        elif args.datasets == "predictions":
            output_dir = Path(args.output_data)
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            if not download_from_huggingface("predictions", output_dir, PROJECT_ROOT):
                success = False
        else:
            output_dir = Path(args.output_data)
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            if not download_from_huggingface(args.datasets, output_dir, PROJECT_ROOT):
                success = False

    if args.model:
        out_models = Path(args.output_models)
        if not out_models.is_absolute():
            out_models = PROJECT_ROOT / out_models
        if not download_model(args.model, out_models, PROJECT_ROOT):
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
