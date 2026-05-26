"""Reproducibility metadata: seeds, environment fingerprint, model card.

Why a ``ModelCard``?
    Production ML models must answer four questions on demand:

    1. *What was trained?*  → name, task, target, features, schema groups.
    2. *On what data?*       → row count, column count, SHA-256 fingerprint.
    3. *With what config?*   → hyperparameters, random seed.
    4. *In what environment?* → Python, OS, package versions, IAD version.

    The ``ModelCard`` dataclass captures all four in a JSON-serialisable form
    so it can be persisted next to the model file (``persistence.py``) and
    logged to MLflow (``tracking/mlflow_tracker.py``).
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

from iad.core.logging import get_logger

logger = get_logger("iad.ml.training.reproducibility")


# ---------------------------------------------------------------------------
# Seed manager
# ---------------------------------------------------------------------------
class SeedManager:
    """Set the global random seeds across every library we use.

    sklearn estimators read ``random_state`` from their kwargs so the seed
    propagates there; this manager covers the libraries that read from a
    process-wide RNG (numpy, python random, optional torch / tensorflow).
    """

    @staticmethod
    def set_global_seed(seed: int) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Optional, soft-skip if these libraries are not installed.
        try:  # pragma: no cover — optional dep
            import torch  # type: ignore[import-not-found]

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        try:  # pragma: no cover — optional dep
            import tensorflow as tf  # type: ignore[import-not-found]

            tf.random.set_seed(seed)
        except ImportError:
            pass
        logger.debug("global seed set", extra={"ctx_seed": seed})


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "shap",
    "lime",
    "mlflow",
    "mlflow-skinny",
    "flaml",
    "pandera",
    "streamlit",
)


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Snapshot of the runtime that produced a model."""

    python: str
    platform: str
    machine: str
    packages: dict[str, str] = field(default_factory=dict)


def capture_environment(extra: Iterable[str] = ()) -> EnvironmentFingerprint:
    pkgs: dict[str, str] = {}
    for name in list(_TRACKED_PACKAGES) + list(extra):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return EnvironmentFingerprint(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        packages=pkgs,
    )


# ---------------------------------------------------------------------------
# Dataset fingerprint
# ---------------------------------------------------------------------------
def fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 hex digest for a DataFrame.

    The hash is computed from a deterministic byte representation: column
    names sorted, dtypes serialised, and the row data hashed via pandas'
    own row-level hasher. Two semantically identical frames produce the
    same digest regardless of memory layout.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("fingerprint_dataframe requires a pandas DataFrame")
    sha = hashlib.sha256()
    sha.update(b"shape:")
    sha.update(repr(df.shape).encode("utf-8"))
    sha.update(b"|columns:")
    sha.update("|".join(sorted(df.columns.astype(str))).encode("utf-8"))
    sha.update(b"|dtypes:")
    dtype_str = "|".join(f"{c}:{dt}" for c, dt in zip(sorted(df.columns), df.dtypes.astype(str)))
    sha.update(dtype_str.encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    except TypeError:
        # Fall back to string serialisation for non-hashable dtypes.
        row_hashes = df.astype(str).values.tobytes()
    sha.update(b"|rows:")
    sha.update(row_hashes)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCard:
    """Full reproducibility metadata for a trained pipeline."""

    name: str
    task: str
    target: str
    features: list[str]
    schema_groups: dict[str, list[str]]
    metrics: dict[str, float]
    cv_metrics: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    dataset_fingerprint: str = ""
    n_rows: int = 0
    n_columns: int = 0
    iad_version: str = "0.0.0"
    environment: EnvironmentFingerprint = field(
        default_factory=lambda: EnvironmentFingerprint(python="", platform="", machine="")
    )
    train_time_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    notes: str = ""
    mlflow_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["environment"] = asdict(self.environment)
        return d
