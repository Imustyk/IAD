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


def test_training_config_custom_primary_metric() -> None:
    config = TrainingConfig(task="classification", primary_metric="accuracy")
    assert config.primary() == "accuracy"


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


def test_training_service_explicit_feature_columns(iris_dataframe: pd.DataFrame) -> None:
    features = [c for c in iris_dataframe.columns if c != "species"]
    service = TrainingService()
    config = TrainingConfig(
        task="classification",
        selected_models=["Random Forest"],
        cross_validate_best=False,
    )
    result = service.train(
        iris_dataframe,
        target="species",
        feature_columns=features,
        config=config,
    )
    assert result.best_entry.model_name == "Random Forest"


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


def test_validate_task_target_allows_classification_labels() -> None:
    from iad.ml.training.service import _validate_task_target

    _validate_task_target("classification", "label", pd.Series(["a", "b", "c"], dtype="str"))
    _validate_task_target("classification", "label", pd.Series(["a", "b"], dtype=object))
    _validate_task_target("classification", "flag", pd.Series([True, False]))
    _validate_task_target(
        "classification",
        "label",
        pd.Series(pd.Categorical(["x", "y"])),
    )
    _validate_task_target("regression", "y", pd.Series([1.0, 2.0, 3.0]))


def test_training_service_rejects_empty_features(iris_dataframe: pd.DataFrame) -> None:
    service = TrainingService()
    config = TrainingConfig(task="classification")
    with pytest.raises(TrainingError, match="at least one feature"):
        service.train(
            iris_dataframe,
            target="species",
            config=config,
            feature_columns=[],
        )


def test_training_service_rejects_regression_on_string_target() -> None:
    from src.data_loader import _load_wine

    df = _load_wine()
    service = TrainingService()
    config = TrainingConfig(task="regression", cross_validate_best=False)
    with pytest.raises(TrainingError, match="non-numeric") as exc_info:
        service.train(df, target="wine_class", config=config)
    assert "classification" in exc_info.value.user_message.lower()


def test_training_service_unknown_model_raises(iris_dataframe: pd.DataFrame) -> None:
    service = TrainingService()
    config = TrainingConfig(task="classification", selected_models=["Nope"])
    with pytest.raises(TrainingError):
        service.train(iris_dataframe, target="species", config=config)
