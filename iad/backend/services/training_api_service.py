"""Training API orchestration — wraps :class:`iad.ml.training.service.TrainingService`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.schemas.ml import LeaderboardRowOut, TrainRequest, TrainResponse
from iad.backend.services.dataframe_io import load_dataframe_from_bytes
from iad.backend.services.persistence_service import PersistenceService
from iad.config.settings import get_settings
from iad.core.exceptions import NotFoundError, TrainingError, ValidationError
from iad.core.logging import get_logger
from iad.core.observability.prometheus import inc_training_job, observe_ml_operation
from iad.ml.training.persistence import save_bundle
from iad.ml.training.registry import Task
from iad.ml.training.service import TrainingConfig, TrainingService

logger = get_logger("iad.backend.training_api")


def _infer_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target]
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or str(series.dtype).startswith("category")
    ):
        return "classification"
    nunique = int(series.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(series) and nunique <= 20:
        return "classification"
    if nunique <= 2:
        return "classification"
    return "regression"


class TrainingAPIService:
    """Application service for ``POST /train``."""

    def __init__(
        self,
        training: TrainingService | None = None,
        persistence: PersistenceService | None = None,
    ) -> None:
        self._training = training or TrainingService()
        self._persistence = persistence or PersistenceService()

    def _load_dataframe(
        self,
        request: TrainRequest,
        *,
        session: Session | None,
        user_email: str | None,
    ) -> tuple[pd.DataFrame, str | None]:
        if request.records is not None:
            return pd.DataFrame(request.records), None

        if request.dataset_id:
            if session is None:
                raise ValidationError(
                    "dataset_id requires database enabled",
                    user_message="Dataset lookup requires database integration.",
                )
            uow = UnitOfWork(session)
            record = uow.datasets.get_or_raise(request.dataset_id)
            if not record.storage_path:
                raise ValidationError(
                    "Dataset has no storage_path",
                    user_message="Dataset file is missing on disk.",
                )
            path = Path(record.storage_path)
            if not path.exists():
                raise NotFoundError(
                    f"File missing: {path}",
                    user_message="Dataset file not found on server.",
                )
            data = path.read_bytes()
            df = load_dataframe_from_bytes(data, path.name)
            return df, record.id

        raise ValidationError(
            "Provide either records or dataset_id",
            user_message="Send inline records or a dataset_id from POST /upload.",
        )

    def train(
        self,
        request: TrainRequest,
        *,
        session: Session | None = None,
        user_email: str | None = None,
    ) -> TrainResponse:
        inc_training_job(outcome="started")
        try:
            df, dataset_id = self._load_dataframe(request, session=session, user_email=user_email)
            if request.target_column not in df.columns:
                raise ValidationError(
                    f"Target {request.target_column!r} not in dataset",
                    user_message=f"Target column '{request.target_column}' was not found.",
                )

            task_str = request.task_type or _infer_task_type(df, request.target_column)
            task: Task = "classification" if task_str == "classification" else "regression"

            config = TrainingConfig(
                task=task,
                test_size=request.test_size,
                cv_folds=request.cv_folds,
                selected_models=request.selected_models,
                cross_validate_best=request.cross_validate_best,
                track_mlflow=request.track_mlflow,
                mlflow_experiment=request.mlflow_experiment,
                notes=request.notes,
            )

            result = self._training.train(
                df,
                target=request.target_column,
                config=config,
                feature_columns=request.feature_columns,
            )

            artifact_path: str | None = None
            if request.save_artifact and result.model_card is not None:
                path = save_bundle(result.best_pipeline, result.model_card)
                artifact_path = str(path)

            experiment_id: str | None = None
            champion_model_id: str | None = None
            if request.persist_to_database:
                persisted = self._persistence.persist_training_result(
                    result,
                    experiment_name=request.experiment_name,
                    dataset_id=dataset_id,
                    artifact_path=artifact_path,
                    user_email=user_email,
                )
                experiment_id = persisted.experiment_id
                champion_model_id = persisted.champion_model_id

            leaderboard = [
                LeaderboardRowOut(
                    model_name=e.model_name,
                    family=e.family,
                    metrics=e.metrics,
                    cv_metrics=e.cv_metrics,
                    train_time_seconds=e.train_time_seconds,
                    error=e.error,
                )
                for e in result.leaderboard
            ]

            inc_training_job(outcome="completed")
            observe_ml_operation(operation="train", outcome="success", duration_seconds=0.0)

            return TrainResponse(
                experiment_id=experiment_id,
                champion_model_id=champion_model_id,
                artifact_path=artifact_path,
                task_type=result.task,
                target_column=result.target,
                best_model_name=result.best_entry.model_name,
                metrics=dict(result.best_entry.metrics),
                cv_metrics=dict(result.best_entry.cv_metrics),
                leaderboard=leaderboard,
                features=list(result.features),
            )
        except TrainingError:
            inc_training_job(outcome="failed")
            observe_ml_operation(operation="train", outcome="error")
            raise
        except Exception:
            inc_training_job(outcome="failed")
            observe_ml_operation(operation="train", outcome="error")
            raise
