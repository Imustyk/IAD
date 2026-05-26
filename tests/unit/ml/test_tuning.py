"""Optuna search."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.core.exceptions import TrainingError
from iad.ml.tuning import OptunaSearch, has_search_space


@pytest.fixture(scope="module")
def iris_xy() -> tuple[pd.DataFrame, pd.Series]:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    return df.drop(columns=["species"]), df["species"]


def test_has_search_space_for_known_models() -> None:
    assert has_search_space("Random Forest")
    assert has_search_space("Logistic Regression")
    assert not has_search_space("NoSuch")


def test_optuna_search_random_forest(iris_xy) -> None:
    X, y = iris_xy
    search = OptunaSearch(
        model_name="Random Forest",
        task="classification",
        cv=3,
        n_trials=4,
        random_state=0,
    )
    result = search.fit(X, y)
    assert result.n_trials >= 1
    assert result.best_score >= 0.0
    assert result.best_estimator is not None
    assert "n_estimators" in result.best_params


def test_optuna_search_logistic_regression(iris_xy) -> None:
    X, y = iris_xy
    search = OptunaSearch(
        model_name="Logistic Regression",
        task="classification",
        cv=3,
        n_trials=4,
        random_state=0,
    )
    result = search.fit(X, y)
    assert result.best_estimator.predict(X.head(3)).shape == (3,)


def test_optuna_search_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        OptunaSearch(model_name="Nope", task="classification")


def test_optuna_search_no_search_space_raises() -> None:
    # Synthetic case: register a model without a search space.
    from sklearn.dummy import DummyClassifier

    from iad.ml.training.registry import ModelRegistry, ModelSpec

    reg = ModelRegistry.default()
    reg.register(
        ModelSpec(
            name="Untunable",
            task="classification",
            family="dummy",
            factory=lambda **kw: DummyClassifier(strategy="most_frequent", **kw),
        )
    )
    with pytest.raises(TrainingError):
        OptunaSearch(model_name="Untunable", task="classification", registry=reg)
