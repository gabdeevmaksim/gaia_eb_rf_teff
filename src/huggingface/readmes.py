"""Upload README files to HuggingFace dataset and model repos."""

from pathlib import Path

from . import HF_DATASET_REPO, HF_MODEL_REPO

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def upload_readmes_impl(project_root: Path) -> None:
    """
    Upload HF_DATASET_README.md and HF_MODEL_README.md from project_root/docs
    to the Hub as README.md in each repo.
    """
    if HfApi is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")
    api = HfApi()

    docs_dir = project_root / "docs"
    dataset_readme = docs_dir / "HF_DATASET_README.md"
    model_readme = docs_dir / "HF_MODEL_README.md"

    if not dataset_readme.exists():
        print(f"  ⚠ Not found: {dataset_readme}")
    else:
        print("Uploading dataset README...")
        try:
            api.upload_file(
                path_or_fileobj=str(dataset_readme),
                path_in_repo="README.md",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message="Add comprehensive dataset documentation",
            )
            print("  ✓ Dataset README uploaded")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not model_readme.exists():
        print(f"  ⚠ Not found: {model_readme}")
    else:
        print("\nUploading model README...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_readme),
                path_in_repo="README.md",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                commit_message="Add comprehensive model documentation",
            )
            print("  ✓ Model README uploaded")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n✓ README upload complete")
    print(f"  Dataset: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"  Models:  https://huggingface.co/{HF_MODEL_REPO}")
