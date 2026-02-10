#!/usr/bin/env python3
"""Upload README files (HF_DATASET_README.md, HF_MODEL_README.md) to HuggingFace repos. Implementation in src.huggingface."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.huggingface import upload_readmes


def main():
    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. Set with: export HF_TOKEN=your_token")
        sys.exit(1)
    upload_readmes(PROJECT_ROOT)


if __name__ == "__main__":
    main()
