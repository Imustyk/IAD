"""Bundle save / load."""
from __future__ import annotations

import pytest

from iad.ml.training import (
    TrainingConfig,
    TrainingService,
    load_bundle,
    save_bundle,
)


def test_save_and_load_bundle(tmp_path) -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    service = TrainingService()
    config = TrainingConfig(
        task="classification",
        selected_models=["Logistic Regression"],
        cv_folds=3,
        cross_validate_best=False,
    )
    result = service.train(df, target="species", config=config)
    path = tmp_path / "iris_logistic.joblib"
    save_bundle(result.best_pipeline, result.model_card, path=path)
    assert path.exists()
    pipeline, card = load_bundle(path, trusted_path_only=False)
    preds = pipeline.predict(df.head(5).drop(columns=["species"]))
    assert len(preds) == 5
    assert card.name == "Logistic Regression"
    assert card.task == "classification"
    assert card.dataset_fingerprint


def test_save_bundle_refuses_overwrite(tmp_path) -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    config = TrainingConfig(
        task="classification",
        selected_models=["Logistic Regression"],
        cv_folds=3,
        cross_validate_best=False,
    )
    result = TrainingService().train(df, target="species", config=config)
    path = tmp_path / "iris.joblib"
    save_bundle(result.best_pipeline, result.model_card, path=path)
    with pytest.raises(Exception):
        save_bundle(result.best_pipeline, result.model_card, path=path, overwrite=False)
    save_bundle(result.best_pipeline, result.model_card, path=path, overwrite=True)
