"""Tests for training service bridge (no Streamlit)."""
from __future__ import annotations

import pytest
from sklearn.datasets import load_iris

from iad.frontend.services.training import train_legacy


@pytest.mark.slow
def test_train_legacy_iris() -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    pipeline, report = train_legacy(
        df=df,
        target="species",
        feature_columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        task_type="classification",
        test_size=0.2,
        random_state=42,
        cv_folds=3,
        selected_models=["Logistic Regression", "Random Forest"],
    )
    assert pipeline is not None
    assert report.engine == "legacy"
    assert not report.leaderboard.empty
    assert report.best_model_name
