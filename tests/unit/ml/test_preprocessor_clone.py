"""Ensure each leaderboard candidate gets an independent preprocessor."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from iad.ml.training.registry import ModelRegistry, ModelSpec
from iad.ml.training.service import TrainingConfig, TrainingService
from src.data_loader import load_sample


def test_train_single_clones_preprocessor() -> None:
    registry = ModelRegistry()
    registry.register(
        ModelSpec(name="A", task="classification", family="linear", factory=LogisticRegression)
    )
    registry.register(
        ModelSpec(name="B", task="classification", family="linear", factory=LogisticRegression)
    )
    svc = TrainingService(registry=registry)
    df = load_sample("Iris (classification)")
    config = TrainingConfig(
        task="classification",
        selected_models=["A", "B"],
        cross_validate_best=False,
        cv_folds=3,
    )
    result = svc.train(df, target="species", config=config)
    assert result.best_entry.model_name in {"A", "B"}
    assert result.best_pipeline is not None
