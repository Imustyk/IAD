"""Persistence + training integration."""
from __future__ import annotations

import pytest

from iad.backend.services.persistence_service import PersistenceService
from iad.ml.training.registry import ModelRegistry
from iad.ml.training.service import TrainingConfig, TrainingService


@pytest.mark.integration
def test_persist_training_result(iris_df, db_session) -> None:
    target = "species"
    features = [c for c in iris_df.columns if c != target]
    service = TrainingService(ModelRegistry.default())
    result = service.train(
        iris_df,
        target=target,
        feature_columns=features,
        config=TrainingConfig(
            task="classification",
            test_size=0.25,
            cv_folds=3,
            random_state=42,
            selected_models=["Logistic Regression"],
        ),
    )
    persisted = PersistenceService().persist_training_result(
        result,
        experiment_name="pytest-exp",
        user_email="persist@example.com",
    )
    assert persisted.experiment_id
    assert persisted.metric_count >= 1
