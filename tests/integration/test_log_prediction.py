"""Prediction audit log integration."""
from __future__ import annotations

import uuid

import pytest

from iad.backend.services.persistence_service import PersistenceService
from iad.ml.training.registry import ModelRegistry
from iad.ml.training.service import TrainingConfig, TrainingService


@pytest.mark.integration
def test_log_prediction_after_training(iris_df, db_session) -> None:
    target = "species"
    features = [c for c in iris_df.columns if c != target]
    result = TrainingService(ModelRegistry.default()).train(
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
        experiment_name=f"pred-{uuid.uuid4().hex[:6]}",
        user_email="pred@example.com",
    )
    assert persisted.champion_model_id
    pred_id = PersistenceService().log_prediction(
        model_id=persisted.champion_model_id,
        input_row={features[0]: float(iris_df[features[0]].iloc[0])},
        output="setosa",
        latency_ms=12.5,
        user_email="pred@example.com",
    )
    assert pred_id
