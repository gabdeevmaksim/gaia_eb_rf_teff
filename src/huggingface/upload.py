"""Upload datasets and models to HuggingFace Hub; clean repos and create repos."""

from pathlib import Path

import yaml

from . import HF_DATASET_REPO, HF_MODEL_REPO

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None
    create_repo = None


def create_repositories_impl() -> None:
    """Create dataset and model repos if they don't exist."""
    if create_repo is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")
    print("Creating HuggingFace repositories (if they don't exist)...")
    for repo_id, repo_type, label in [
        (HF_DATASET_REPO, "dataset", "Dataset"),
        (HF_MODEL_REPO, "model", "Model"),
    ]:
        try:
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
                private=False,
            )
            url = (
                f"https://huggingface.co/datasets/{repo_id}"
                if repo_type == "dataset"
                else f"https://huggingface.co/{repo_id}"
            )
            print(f"  ✓ {label}: {url}")
        except Exception as e:
            print(f"  ✗ Error creating {label} repo: {e}")


def upload_datasets_impl(dataset_type: str, project_root: Path) -> None:
    """Upload datasets. project_root is used for resolving data/processed and config."""
    if HfApi is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")
    api = HfApi()

    datasets = {
        "photometry": {
            "folder": project_root / "data" / "raw",
            "patterns": ["eb_unified_photometry.parquet", "eb_unified_photometry_SUMMARY.txt"],
            "path_in_repo": "photometry",
        },
        "predictions": {
            "folder": project_root / "data" / "processed",
            "patterns": [
                "eb_catalog_teff.parquet",
                "eb_catalog_teff_SUMMARY.txt",
            ],
            "path_in_repo": "predictions",
        },
        "catalog": {
            "folder": project_root / "data" / "processed",
            "patterns": ["stars_types_with_best_predictions*"],
            "path_in_repo": "catalogs",
        },
        "training": {
            "folder": project_root / "data" / "processed",
            "patterns": ["gaia_all_colors_train*.parquet"],
            "path_in_repo": "training",
        },
    }

    if dataset_type == "all":
        for dt in ["photometry", "predictions", "catalog", "training"]:
            upload_datasets_impl(dt, project_root)
        return

    if dataset_type not in datasets:
        print(f"Error: Unknown dataset type: {dataset_type}")
        return

    info = datasets[dataset_type]
    folder = info["folder"]
    if not folder.exists():
        print(f"Error: Data folder not found: {folder}")
        return

    print(f"\nUploading {dataset_type} datasets to {HF_DATASET_REPO}...")
    for pattern in info["patterns"]:
        matching = list(folder.glob(pattern))
        if not matching:
            print(f"  ⚠ No files matching pattern: {pattern}")
            continue
        for fp in matching:
            try:
                print(f"  Uploading: {fp.name} ({fp.stat().st_size / 1024 / 1024:.1f} MB)")
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=f"{info['path_in_repo']}/{fp.name}",
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                )
                print("    ✓ Uploaded successfully")
            except Exception as e:
                print(f"    ✗ Error: {e}")
    print(f"\n✓ {dataset_type.capitalize()} datasets uploaded")


def _upload_single_model(
    api: HfApi,
    model_path: Path,
    display_name: str,
) -> None:
    """Upload one .pkl and its _metadata.json / _SUMMARY.txt if present."""
    name = model_path.name
    print(f"\nUploading: {display_name}")
    print(f"  File: {name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=name,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    print(f"    ✓ {name}")
    for suffix in ("_metadata.json", "_SUMMARY.txt"):
        other = model_path.with_name(model_path.stem + suffix)
        if other.exists():
            api.upload_file(
                path_or_fileobj=str(other),
                path_in_repo=other.name,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print(f"    ✓ {other.name}")


def upload_models_impl(
    model_name: str,
    project_root: Path,
    *,
    models_from_dir: bool = False,
) -> None:
    """Upload models. project_root is used for models/ and config/models/model_registry.yaml."""
    if HfApi is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")
    api = HfApi()
    models_dir = project_root / "models"
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    if model_name == "all" and models_from_dir:
        pkl_files = [
            p
            for p in models_dir.glob("*.pkl")
            if "_clustering_" not in p.stem and "_scaler" not in p.stem
        ]
        if not pkl_files:
            print("No .pkl models found in models/ (excluding _clustering_ / _scaler)")
            return
        print(f"\nUploading {len(pkl_files)} model(s) from models/ to {HF_MODEL_REPO}...")
        for p in sorted(pkl_files):
            _upload_single_model(api, p, p.stem)
        print("\n✓ All models from directory uploaded")
        return

    registry_path = project_root / "config" / "models" / "model_registry.yaml"
    if not registry_path.exists():
        if model_name == "all":
            print("No model_registry.yaml; use --models-from-dir to upload all .pkl from models/")
            return
        print(f"Error: Model registry not found at {registry_path}")
        return

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    if model_name == "all":
        for key in registry.get("models", {}).keys():
            upload_models_impl(key, project_root, models_from_dir=False)
        try:
            print("\nUploading model registry...")
            api.upload_file(
                path_or_fileobj=str(registry_path),
                path_in_repo="model_registry.yaml",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print("  ✓ Model registry uploaded")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        return

    if model_name.endswith(".pkl"):
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return
        _upload_single_model(api, model_path, model_name)
        return

    if model_name not in registry.get("models", {}):
        print(f"Error: Model '{model_name}' not found in registry (use a key or a .pkl filename)")
        return

    model_file = registry["models"][model_name]["file"]
    model_path = models_dir / model_file
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    _upload_single_model(api, model_path, model_name)
