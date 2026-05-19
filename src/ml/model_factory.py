from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    backend: str  # "sklearn" | "xgboost" | "cuml"
    name: str


def get_model_spec(model_config: Dict[str, Any]) -> ModelSpec:
    model_block = model_config.get("model", {}) or {}
    backend = (model_block.get("backend") or "sklearn").strip().lower()
    name = model_block.get("name") or backend
    return ModelSpec(backend=backend, name=name)


def build_regressor(
    *,
    model_config: Dict[str, Any],
    hyperparameters: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a regressor from config.

    Returns
    -------
    (model, resolved_hyperparameters)
    """
    spec = get_model_spec(model_config)

    if spec.backend == "sklearn":
        from sklearn.ensemble import RandomForestRegressor

        # Keep current RF defaults/keys unchanged (backward compatible).
        max_features = hyperparameters.get("max_features", "log2")
        if isinstance(max_features, str) and max_features.isdigit():
            max_features = int(max_features)

        params = {
            "n_estimators": hyperparameters.get("n_estimators", 300),
            "max_depth": hyperparameters.get("max_depth", 20),
            "min_samples_split": hyperparameters.get("min_samples_split", 5),
            "min_samples_leaf": hyperparameters.get("min_samples_leaf", 4),
            "max_features": max_features,
            "random_state": hyperparameters.get("random_state", 42),
            "n_jobs": hyperparameters.get("n_jobs", -1),
            "verbose": hyperparameters.get("verbose", 0),
        }
        return RandomForestRegressor(**params), params

    if spec.backend == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost backend requested but xgboost is not installed. "
                "Install with: pip install xgboost"
            ) from e

        # GPU on by default (can be overridden in hyperparameters).
        tree_method = hyperparameters.get("tree_method", "gpu_hist")
        predictor = hyperparameters.get("predictor", "gpu_predictor")
        device = hyperparameters.get("device")  # optional (newer xgboost)

        params: Dict[str, Any] = {
            "n_estimators": hyperparameters.get("n_estimators", 1000),
            "max_depth": hyperparameters.get("max_depth", 8),
            "learning_rate": hyperparameters.get("learning_rate", 0.05),
            "subsample": hyperparameters.get("subsample", 0.9),
            "colsample_bytree": hyperparameters.get("colsample_bytree", 0.9),
            "min_child_weight": hyperparameters.get("min_child_weight", 1.0),
            "reg_lambda": hyperparameters.get("reg_lambda", 1.0),
            "reg_alpha": hyperparameters.get("reg_alpha", 0.0),
            "gamma": hyperparameters.get("gamma", 0.0),
            "objective": hyperparameters.get("objective", "reg:squarederror"),
            "random_state": hyperparameters.get("random_state", 42),
            "n_jobs": hyperparameters.get("n_jobs", -1),
            "tree_method": tree_method,
            "predictor": predictor,
        }
        if device is not None:
            params["device"] = device

        model = xgb.XGBRegressor(**params)
        return model, params

    if spec.backend == "cuml":
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
        except ImportError as e:
            raise ImportError(
                "cuML backend requested but RAPIDS cuML is not installed in this environment."
            ) from e

        # cuML params are similar but not identical; keep a conservative subset.
        # Note: cuML trains on GPU; n_streams and other advanced options are omitted.
        params = {
            "n_estimators": hyperparameters.get("n_estimators", 500),
            "max_depth": hyperparameters.get("max_depth", 20),
            "max_features": hyperparameters.get("max_features", 1.0),
            "random_state": hyperparameters.get("random_state", 42),
        }
        return CumlRF(**params), params

    raise ValueError(f"Unknown model backend: {spec.backend!r} (expected sklearn|xgboost|cuml)")

