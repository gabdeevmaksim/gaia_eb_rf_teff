#!/usr/bin/env python3
"""
Upload datasets and models to HuggingFace Hub.

Implementation lives in src.huggingface. Run from project root.

Usage:
    python scripts/upload_to_huggingface.py --clean --datasets all --models all
    python scripts/upload_to_huggingface.py --datasets photometry
    python scripts/upload_to_huggingface.py --datasets predictions
    python scripts/upload_to_huggingface.py --models all --models-from-dir
"""

import argparse
import os
import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.huggingface import (
    HF_DATASET_REPO,
    HF_MODEL_REPO,
    clean_repo,
    create_repositories,
    upload_datasets,
    upload_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Upload datasets and models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        choices=["photometry", "predictions", "catalog", "training", "all"],
        help="Dataset type to upload",
    )
    parser.add_argument(
        "--models",
        help="Model name (registry key, .pkl filename, or 'all')",
    )
    parser.add_argument(
        "--models-from-dir",
        action="store_true",
        help="With --models all: upload all .pkl from models/ (skip clustering/scaler)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all existing files from the target repo(s) before uploading",
    )
    parser.add_argument(
        "--create-repos",
        action="store_true",
        help="Create HuggingFace repositories if they don't exist",
    )

    args = parser.parse_args()

    if not any([args.datasets, args.models, args.create_repos, args.clean]):
        parser.print_help()
        print("\nError: Specify --datasets, --models, --create-repos, or --clean")
        sys.exit(1)

    if not os.getenv("HF_TOKEN") and (args.datasets or args.models or args.clean):
        print("Warning: HF_TOKEN not set. Use: huggingface-cli login")
        if input("Continue? (y/N): ").lower() != "y":
            sys.exit(1)

    if args.create_repos:
        create_repositories()

    if args.clean:
        print("\nCleaning HuggingFace repositories...")
        if args.datasets:
            clean_repo(HF_DATASET_REPO, "dataset", commit_message="Clean dataset repo before re-upload")
        if args.models:
            clean_repo(HF_MODEL_REPO, "model", commit_message="Clean model repo before re-upload")
        if not args.datasets and not args.models:
            clean_repo(HF_DATASET_REPO, "dataset", commit_message="Clean dataset repo")
            clean_repo(HF_MODEL_REPO, "model", commit_message="Clean model repo")

    if args.datasets:
        upload_datasets(args.datasets, PROJECT_ROOT)

    if args.models:
        upload_models(args.models, PROJECT_ROOT, models_from_dir=args.models_from_dir)

    print("\n✓ Upload complete!")
    print(f"  Datasets: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"  Models:   https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
