"""Training bundle serialization (cloudpickle)."""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from iad.ml.training.serialize import (
    build_training_bundle,
    bundle_from_bytes,
    bundle_to_bytes,
    serialize_training_artifact,
)


def test_bundle_roundtrip_with_custom_pipeline() -> None:
    from iad.ml.preprocessing.transformers.rare_category import RareCategoryGrouper

    pipe = Pipeline(
        [
            ("rare", RareCategoryGrouper(columns=None, min_frequency=0.01)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    bundle = build_training_bundle(
        pipe,
        task_type="classification",
        target="y",
        features=["x"],
        best_model_name="Logistic Regression",
    )
    loaded = bundle_from_bytes(bundle_to_bytes(bundle))
    assert loaded["best_model_name"] == "Logistic Regression"
    assert isinstance(loaded["pipeline"], Pipeline)
    assert "rare" in loaded["pipeline"].named_steps


def test_serialize_training_artifact_bytes() -> None:
    pipe = Pipeline([("model", LogisticRegression(max_iter=200))])
    data = serialize_training_artifact(
        pipe,
        task_type="classification",
        target="species",
        features=["a"],
        best_model_name="Logistic Regression",
    )
    assert isinstance(data, bytes)
    assert bundle_from_bytes(data)["target"] == "species"
