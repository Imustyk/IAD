"""Reproducibility utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from iad.ml.training.reproducibility import (
    EnvironmentFingerprint,
    ModelCard,
    SeedManager,
    capture_environment,
    fingerprint_dataframe,
)


def test_seed_manager_sets_python_numpy_seeds() -> None:
    SeedManager.set_global_seed(123)
    a = random.random()
    b = float(np.random.random())
    SeedManager.set_global_seed(123)
    assert random.random() == a
    assert float(np.random.random()) == b
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_capture_environment_includes_core_packages() -> None:
    env = capture_environment()
    assert isinstance(env, EnvironmentFingerprint)
    assert env.python
    assert env.platform
    # numpy is always installed
    assert "numpy" in env.packages


def test_fingerprint_dataframe_is_deterministic() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    h1 = fingerprint_dataframe(df)
    h2 = fingerprint_dataframe(df.copy())
    assert h1 == h2
    assert len(h1) == 64


def test_fingerprint_dataframe_changes_on_data_change() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    df_changed = pd.DataFrame({"a": [1, 2, 4]})
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df_changed)


def test_model_card_serialises_to_dict() -> None:
    card = ModelCard(
        name="XGBoost",
        task="classification",
        target="churn",
        features=["age", "tenure"],
        schema_groups={"numeric": ["age", "tenure"], "categorical": [], "datetime": [], "target_encoded": []},
        metrics={"f1_macro": 0.81},
        seed=42,
        dataset_fingerprint="abcd",
        n_rows=500,
        n_columns=2,
    )
    d = card.to_dict()
    assert d["name"] == "XGBoost"
    assert d["environment"]["python"] is not None or d["environment"]["python"] == ""
