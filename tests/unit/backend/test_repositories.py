"""Repository unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.services.persistence_service import PersistenceService
from iad.ml.training.registry import ModelRegistry
from iad.ml.training.service import TrainingConfig, TrainingService


def test_user_get_or_create_default(db_session) -> None:
    uow = UnitOfWork(db_session)
    user = uow.users.get_or_create_default("test@example.com")
    again = uow.users.get_or_create_default("test@example.com")
    assert user.id == again.id


def test_dataset_versioning(db_session) -> None:
    uow = UnitOfWork(db_session)
    user = uow.users.create(email="data@example.com")
    v1 = uow.datasets.create_version(
        user_id=user.id,
        name="Telco Churn",
        row_count=100,
        column_count=10,
    )
    v2 = uow.datasets.create_version(
        user_id=user.id,
        name="Telco Churn",
        row_count=120,
        column_count=10,
    )
    assert v1.slug == v2.slug
    assert v2.version == 2


def test_experiment_lifecycle(db_session) -> None:
    uow = UnitOfWork(db_session)
    user = uow.users.create(email="exp@example.com")
    exp = uow.experiments.create(
        user_id=user.id,
        name="run-1",
        task_type="classification",
        target_column="y",
        feature_columns=["a", "b"],
    )
    uow.experiments.mark_running(exp.id)
    uow.experiments.mark_completed(exp.id)
    loaded = uow.experiments.get_or_raise(exp.id)
    assert loaded.status == "completed"
    assert loaded.started_at is not None


def test_champion_model(db_session) -> None:
    uow = UnitOfWork(db_session)
    user = uow.users.create(email="ml@example.com")
    exp = uow.experiments.create(
        user_id=user.id,
        name="run",
        task_type="regression",
        target_column="y",
        feature_columns=["x"],
    )
    m1 = uow.models.create(
        user_id=user.id,
        experiment_id=exp.id,
        name="Model A",
        task_type="regression",
        is_champion=True,
    )
    m2 = uow.models.create(
        user_id=user.id,
        experiment_id=exp.id,
        name="Model B",
        task_type="regression",
        is_champion=True,
    )
    db_session.refresh(m1)
    assert m1.is_champion is False
    assert m2.is_champion is True


def test_register_dataset_service(db_session) -> None:
    svc = PersistenceService(default_user_email="svc@example.com")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = svc.register_dataset(df, name="Demo", source="test")
    assert result.version == 1
    assert result.dataset_id


@pytest.mark.slow
def test_persist_training_result_integration(db_session) -> None:
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})

    service = TrainingService(ModelRegistry.default())
    config = TrainingConfig(
        task="classification",
        test_size=0.25,
        cv_folds=3,
        random_state=0,
        selected_models=["Logistic Regression"],
    )
    result = service.train(
        df,
        target="species",
        config=config,
        feature_columns=[c for c in df.columns if c != "species"],
    )

    persist = PersistenceService(default_user_email="train@example.com")
    out = persist.persist_training_result(result, experiment_name="iris-benchmark")
    assert out.champion_model_id
    assert out.metric_count > 0
