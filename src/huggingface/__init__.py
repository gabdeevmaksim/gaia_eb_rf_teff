"""
HuggingFace Hub integration: upload/download datasets and models, clean repos, upload READMEs.

Use from project root. Scripts in scripts/ provide the CLI entry points.
"""

from pathlib import Path

# Repository IDs (used by upload, download, readmes)
HF_DATASET_REPO = "Dedulek/gaia-eb-teff-datasets"
HF_MODEL_REPO = "Dedulek/gaia-eb-teff-models"

__all__ = [
    "HF_DATASET_REPO",
    "HF_MODEL_REPO",
    "clean_repo",
    "create_repositories",
    "upload_datasets",
    "upload_models",
    "upload_readmes",
    "download_from_huggingface",
    "download_model",
]


def _ensure_hf():
    try:
        from huggingface_hub import HfApi
        return HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e


def clean_repo(
    repo_id: str,
    repo_type: str,
    *,
    commit_message: str = "Clean repo before re-upload",
):
    """
    Delete all files in a HuggingFace repo (dataset or model).

    Parameters
    ----------
    repo_id : str
        Repository ID (e.g. Dedulek/gaia-eb-teff-datasets).
    repo_type : str
        'dataset' or 'model'.
    commit_message : str
        Commit message for the deletion commit.
    """
    api = _ensure_hf()()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"  ⚠ Could not list files (repo may be empty): {e}")
        return
    if not files:
        print("  Repo already empty.")
        return
    print(f"  Deleting {len(files)} file(s) in {repo_id}...")
    api.delete_files(
        repo_id=repo_id,
        delete_patterns=["*"],
        repo_type=repo_type,
        commit_message=commit_message,
    )
    print(f"  ✓ Cleaned {repo_id}")


def create_repositories():
    """Create HuggingFace dataset and model repositories if they don't exist."""
    from .upload import create_repositories_impl

    create_repositories_impl()


def upload_datasets(dataset_type: str, project_root: Path):
    """
    Upload datasets to HuggingFace Hub.

    Parameters
    ----------
    dataset_type : str
        One of: 'photometry', 'predictions', 'catalog', 'training', 'all'.
    project_root : Path
        Project root (paths are resolved relative to this).
    """
    from .upload import upload_datasets_impl

    upload_datasets_impl(dataset_type, project_root)


def upload_models(
    model_name: str,
    project_root: Path,
    *,
    models_from_dir: bool = False,
):
    """
    Upload trained models to HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Registry key, exact .pkl filename, or 'all'.
    project_root : Path
        Project root.
    models_from_dir : bool
        If True and model_name=='all', upload all .pkl from models/ (excluding
        clustering/scaler), not only registry.
    """
    from .upload import upload_models_impl

    upload_models_impl(model_name, project_root, models_from_dir=models_from_dir)


def upload_readmes(project_root: Path):
    """
    Upload HF_DATASET_README.md and HF_MODEL_README.md to the Hub as README.md.

    Parameters
    ----------
    project_root : Path
        Project root (README files are expected in this directory).
    """
    from .readmes import upload_readmes_impl

    upload_readmes_impl(project_root)


def download_from_huggingface(
    dataset_name: str,
    output_dir: str | Path,
    project_root: Path,
) -> bool:
    """
    Download dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        One of: 'training', 'catalog', 'correction', 'all'.
    output_dir : str or Path
        Output directory for downloaded files.
    project_root : Path
        Project root (for resolving paths if needed).

    Returns
    -------
    bool
        True if at least one file was downloaded.
    """
    from .download import download_from_huggingface_impl

    return download_from_huggingface_impl(dataset_name, Path(output_dir), project_root)


def download_model(
    model_name: str,
    output_dir: str | Path,
    project_root: Path,
) -> bool:
    """
    Download trained model from HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Registry key, or 'all', or a .pkl filename.
    output_dir : str or Path
        Output directory for models.
    project_root : Path
        Project root (for registry path).

    Returns
    -------
    bool
        True if download succeeded.
    """
    from .download import download_model_impl

    return download_model_impl(model_name, Path(output_dir), project_root)
