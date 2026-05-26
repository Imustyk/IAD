"""Model registry + TrainingService."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from iad.core.exceptions import TrainingError
from iad.ml.training import (
    ModelRegistry,
    ModelSpec,
    TrainingConfig,
    TrainingService,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_default_has_classification_models() -> None:
    reg = ModelRegistry.default()
    names = reg.names("classification")
    assert "Logistic Regression" in names
    assert "Random Forest" in names
    # XGB / LGBM / CatBoost are optional but installed in this venv
    assert any(n in names for n in ("XGBoost", "LightGBM", "CatBoost"))


def test_registry_default_has_regression_models() -> None:
    reg = ModelRegistry.default()
    names = reg.names("regression")
    assert "Linear Regression" in names
    assert "ElasticNet" in names
    assert "Ridge Regression" in names


def test_registry_register_unregister() -> None:
    reg = ModelRegistry()
    spec = ModelSpec(
        name="Dummy",
        task="classification",
        family="linear",
        factory=lambda **kw: object(),
    )
    reg.register(spec)
    assert "Dummy" in reg.names("classification")
    reg.unregister("classification", "Dummy")
    assert "Dummy" not in reg.names("classification")


def test_registry_get_unknown_raises() -> None:
    reg = ModelRegistry.default()
    with pytest.raises(KeyError):
        reg.get("classification", "NoSuch")


# ---------------------------------------------------------------------------
# TrainingService
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def telco_dataframe() -> pd.DataFrame:
    from src.data_loader import load_sample

    return load_sample("Telco churn (classification)")


@pytest.fixture(scope="module")
def iris_dataframe() -> pd.DataFrame:
    from src.data_loader import load_sample

    return load_sample("Iris (classification)")


def test_training_service_classification_end_to_end(iris_dataframe: pd.DataFrame) -> None:
    service = TrainingService()
    config = TrainingConfig(
        task="classification",
        cv_folds=3,
        selected_models=["Logistic Regression", "Random Forest", "Hist Gradient Boosting"],
        cross_validate_best=True,
    )
    result = service.train(iris_dataframe, target="species", config=config)
    assert isinstance(result.best_pipeline, Pipeline)
    assert result.best_entry.metrics["accuracy"] > 0.85
    assert result.confusion_matrix is not None
    assert result.feature_importance is not None
    assert result.model_card is not None
    assert result.model_card.dataset_fingerprint
    leaderboard_df = result.leaderboard_frame()
    assert "model" in leaderboard_df.columns
    assert len(leaderboard_df) >= 3


def test_training_service_regression_end_to_end() -> None:
    from src.data_loader import load_sample

    df = load_sample("Diabetes progression (regression)")
    service = TrainingService()
    config = TrainingConfig(
        task="regression",
        cv_folds=3,
        selected_models=["Linear Regression", "Ridge Regression", "Hist Gradient Boosting"],
        cross_validate_best=False,
    )
    result = service.train(df, target="disease_progression", config=config)
    assert result.regression_report is not None
    assert result.best_entry.metrics["rmse"] > 0
    assert result.model_card.task == "regression"


def test_training_service_rejects_constant_target() -> None:
    df = pd.DataFrame({"x": list(range(50)), "y": [1] * 50})
    service = TrainingService()
    config = TrainingConfig(task="classification")
    with pytest.raises(Exception):
        service.train(df, target="y", config=config)


def test_training_service_unknown_model_raises(iris_dataframe: pd.DataFrame) -> None:
    service = TrainingService()
    config = TrainingConfig(task="classification", selected_models=["Nope"])
    with pytest.raises(TrainingError):
        service.train(iris_dataframe, target="species", config=config)
