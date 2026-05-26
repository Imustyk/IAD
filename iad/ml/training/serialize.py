"""Serialize trained pipelines for download and disk persistence.

Streamlit reloads modules on rerun, which breaks standard pickle/joblib for
custom transformers (class identity changes). ``cloudpickle`` captures the
bundle at fit time; consumers load bytes instead of re-pickling live objects.
"""
from __future__ import annotations

import io
from typing import Any

import cloudpickle
from sklearn.pipeline import Pipeline

from iad.ml.training.reproducibility import ModelCard

_BUNDLE_VERSION = 1


def build_training_bundle(
    pipeline: Pipeline,
    *,
    task_type: str,
    target: str,
    features: list[str],
    best_model_name: str,
    model_card: ModelCard | None = None,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "version": _BUNDLE_VERSION,
        "pipeline": pipeline,
        "task_type": task_type,
        "target": target,
        "features": list(features),
        "best_model_name": best_model_name,
    }
    if model_card is not None:
        bundle["model_card"] = model_card.to_dict()
    return bundle


def bundle_to_bytes(bundle: dict[str, Any]) -> bytes:
    return cloudpickle.dumps(bundle)


def bundle_from_bytes(data: bytes) -> dict[str, Any]:
    return cloudpickle.loads(data)


def bundle_to_buffer(bundle: dict[str, Any]) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(bundle_to_bytes(bundle))
    buffer.seek(0)
    return buffer


def dump_bundle(path: str | Any, bundle: dict[str, Any]) -> None:
    """Write a bundle to disk using cloudpickle (``.joblib`` extension for convention)."""
    with open(path, "wb") as handle:
        cloudpickle.dump(bundle, handle)


def load_bundle_file(path: str | Any) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return cloudpickle.load(handle)


def serialize_training_artifact(
    pipeline: Pipeline,
    *,
    task_type: str,
    target: str,
    features: list[str],
    best_model_name: str,
    model_card: ModelCard | None = None,
) -> bytes:
    """Pickle a bundle immediately after training (safe across Streamlit reruns)."""
    return bundle_to_bytes(
        build_training_bundle(
            pipeline,
            task_type=task_type,
            target=target,
            features=features,
            best_model_name=best_model_name,
            model_card=model_card,
        )
    )
