"""Enterprise training bridge tests."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_train_enterprise_iris(iris_df) -> None:
    from iad.frontend.services.training import train_enterprise

    target = "species"
    features = [c for c in iris_df.columns if c != target]
    pipeline, report = train_enterprise(
        iris_df,
        target=target,
        feature_columns=features,
        task_type="classification",
        test_size=0.25,
        random_state=42,
        cv_folds=3,
        selected_models=["Logistic Regression", "Random Forest"],
        track_mlflow=False,
    )
    assert pipeline is not None
    assert report.engine == "enterprise"
    assert report.best_model_name
    assert not report.leaderboard.empty


@pytest.mark.unit
def test_get_training_service() -> None:
    from iad.frontend.services.training import get_training_service

    svc = get_training_service()
    assert svc is not None
