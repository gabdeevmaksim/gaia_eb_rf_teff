"""Download datasets and models from HuggingFace Hub."""

import shutil
from pathlib import Path

import yaml

from . import HF_DATASET_REPO, HF_MODEL_REPO

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None


def download_from_huggingface_impl(
    dataset_name: str,
    output_dir: Path,
    project_root: Path,
) -> bool:
    """Download dataset files. Returns True if at least one file was downloaded."""
    if hf_hub_download is None or snapshot_download is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")

    datasets = {
        "training": [
            "photometry/eb_unified_photometry.parquet",
            "photometry/eb_unified_photometry_SUMMARY.txt",
        ],
        "catalog": [
            "catalogs/stars_types_with_best_predictions.fits",
            "catalogs/stars_types_with_best_predictions_DESCRIPTION.txt",
        ],
        "correction": [
            "correction/teff_correction_coeffs_deg2.pkl",
        ],
        "all": [
            "photometry/*.parquet",
            "photometry/*.txt",
            "catalogs/*.fits",
            "catalogs/*.txt",
            "correction/*.pkl",
        ],
    }

    files = datasets.get(dataset_name, [dataset_name])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = 0

    print(f"Downloading {dataset_name} from {HF_DATASET_REPO}...")
    for file_pattern in files:
        try:
            print(f"  Downloading: {file_pattern}")
            if "*" in file_pattern:
                snapshot_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    allow_patterns=[file_pattern],
                    local_dir=output_dir,
                )
                downloaded_count += 1
            else:
                filename = Path(file_pattern).name
                target_path = output_dir / filename
                if target_path.exists():
                    print(f"    ✓ Already exists: {target_path} (skipping)")
                    downloaded_count += 1
                    continue
                downloaded_file = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=file_pattern,
                    repo_type="dataset",
                )
                if Path(downloaded_file).resolve() != target_path.resolve():
                    shutil.copy2(downloaded_file, target_path)
                print(f"    ✓ Downloaded to: {target_path}")
                downloaded_count += 1
        except Exception as e:
            print(f"    ✗ Error downloading {file_pattern}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDownloaded {downloaded_count} file(s) to {output_dir}")
    return downloaded_count > 0


def download_model_impl(
    model_name: str,
    output_dir: Path,
    project_root: Path,
) -> bool:
    """Download model file(s). Returns True on success."""
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = project_root / "config" / "models" / "model_registry.yaml"

    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
    else:
        registry = {"models": {}}

    if model_name == "all":
        if not registry.get("models"):
            print("Error: No models defined in registry")
            return False
        for key in registry["models"].keys():
            if not download_model_impl(key, output_dir, project_root):
                return False
        return True

    if model_name in registry.get("models", {}):
        model_file = registry["models"][model_name]["file"]
        try:
            print(f"  Downloading: {model_file}")
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=model_file,
                local_dir=output_dir,
                repo_type="model",
            )
            metadata_file = model_file.replace(".pkl", "_metadata.json")
            print(f"  Downloading: {metadata_file}")
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=metadata_file,
                local_dir=output_dir,
                repo_type="model",
            )
            print(f"\n✓ Model downloaded: {model_name}")
            return True
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return False

    # Direct .pkl filename
    print(f"Trying direct download for: {model_name}")
    try:
        filename = f"{model_name}.pkl" if not model_name.endswith(".pkl") else model_name
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            local_dir=output_dir,
            repo_type="model",
        )
        print(f"✓ Downloaded to {output_dir / filename}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
