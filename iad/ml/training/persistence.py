"""Save / load model bundles with their model cards.

A bundle is a single ``.joblib`` file containing:

* the fitted sklearn pipeline,
* the :class:`iad.ml.training.reproducibility.ModelCard`,
* the task and target name,
* the IAD package version.

Persisting the model card alongside the pipeline guarantees auditability —
two months from now we can look at any saved model and answer "what data,
what code, what hyperparameters produced this?" without consulting MLflow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cloudpickle
import joblib
from sklearn.pipeline import Pipeline

from iad.config.settings import get_settings
from iad.core.exceptions import IADError, UploadError, ValidationError
from iad.core.logging import get_logger
from iad.core.paths import models_dir, resolve_trusted_artifact_path, safe_filename
from iad.ml.training.reproducibility import ModelCard

logger = get_logger("iad.ml.training.persistence")


_BUNDLE_VERSION = 1


def save_bundle(
    pipeline: Pipeline,
    model_card: ModelCard,
    *,
    path: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist a trained pipeline + model card to disk.

    Args:
        pipeline: a fitted sklearn-compatible pipeline.
        model_card: the :class:`ModelCard` describing the model.
        path: explicit destination. ``None`` → ``models/{name}__{timestamp}.joblib``.
        overwrite: when ``False`` and the destination exists, raise.

    Returns:
        Absolute path to the saved bundle.
    """
    if path is None:
        timestamp = model_card.created_at.replace(":", "").replace("-", "")[:15]
        filename = safe_filename(f"{model_card.name}__{timestamp}.joblib")
        target = models_dir() / filename
    else:
        target = Path(path)
    if target.exists() and not overwrite:
        raise IADError(
            f"target file already exists: {target}",
            user_message=f"Bundle already exists at {target.name}. Pass overwrite=True to replace it.",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "version": _BUNDLE_VERSION,
        "pipeline": pipeline,
        "model_card": _serialise_card(model_card),
        "task": model_card.task,
        "target": model_card.target,
        "features": list(model_card.features),
    }
    joblib.dump(bundle, target, pickle_module=cloudpickle)
    logger.info(
        "bundle saved",
        extra={"ctx_path": str(target), "ctx_model": model_card.name, "ctx_target": model_card.target},
    )
    return target


def load_bundle(path: Path | str, *, trusted_path_only: bool = True) -> tuple[Pipeline, ModelCard]:
    """Load a bundle written by :func:`save_bundle` and return its pipeline + card.

    Args:
        path: Path to a ``.joblib`` file.
        trusted_path_only: When True (default), reject paths outside
            ``models/``, ``data/uploads/``, and ``exports/``.
    """
    resolved = resolve_trusted_artifact_path(path) if trusted_path_only else Path(path).resolve()
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = resolved.stat().st_size
    if size > max_bytes:
        raise UploadError(
            f"Bundle size {size} exceeds limit.",
            user_message=f"Model bundle exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
        )
    if resolved.suffix.lower() != ".joblib":
        raise ValidationError(
            "Model bundles must use the .joblib extension.",
            user_message="Only .joblib model bundles are supported.",
        )

    payload = joblib.load(resolved, pickle_module=cloudpickle)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise IADError(
            f"file at {path} is not an IAD bundle",
            user_message="The selected file is not a valid IAD model bundle.",
        )
    pipeline = payload["pipeline"]
    card = _deserialise_card(payload.get("model_card", {}))
    logger.info("bundle loaded", extra={"ctx_path": str(resolved), "ctx_model": card.name})
    return pipeline, card


def _serialise_card(card: ModelCard) -> dict[str, Any]:
    return card.to_dict()


def _deserialise_card(data: dict[str, Any]) -> ModelCard:
    from iad.ml.training.reproducibility import EnvironmentFingerprint

    env_data = data.get("environment", {})
    if isinstance(env_data, EnvironmentFingerprint):
        env = env_data
    else:
        env = EnvironmentFingerprint(
            python=env_data.get("python", ""),
            platform=env_data.get("platform", ""),
            machine=env_data.get("machine", ""),
            packages=env_data.get("packages", {}),
        )
    return ModelCard(
        name=data.get("name", "unknown"),
        task=data.get("task", "classification"),
        target=data.get("target", ""),
        features=list(data.get("features", [])),
        schema_groups=data.get("schema_groups", {}),
        metrics=data.get("metrics", {}),
        cv_metrics=data.get("cv_metrics", {}),
        best_params=data.get("best_params", {}),
        seed=int(data.get("seed", 42)),
        dataset_fingerprint=data.get("dataset_fingerprint", ""),
        n_rows=int(data.get("n_rows", 0)),
        n_columns=int(data.get("n_columns", 0)),
        iad_version=data.get("iad_version", "0.0.0"),
        environment=env,
        train_time_seconds=float(data.get("train_time_seconds", 0.0)),
        created_at=data.get("created_at", ""),
        notes=data.get("notes", ""),
        mlflow_run_id=data.get("mlflow_run_id"),
    )
