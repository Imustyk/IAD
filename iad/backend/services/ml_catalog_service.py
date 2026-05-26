"""Read-only catalog — models, experiments, ML metrics."""
from __future__ import annotations

from sqlalchemy.orm import Session

from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.schemas.ml import (
    ExperimentsListResponse,
    ExperimentSummaryOut,
    MetricOut,
    MetricsListResponse,
    ModelsListResponse,
    ModelSummaryOut,
)
from iad.core.exceptions import NotFoundError, ValidationError
from iad.core.logging import get_logger

logger = get_logger("iad.backend.ml_catalog")


class MLCatalogService:
    """List models, experiments, and stored evaluation metrics."""

    def list_models(
        self,
        *,
        user_id: str,
        session: Session,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> ModelsListResponse:
        uow = UnitOfWork(session)
        if experiment_id:
            exp = uow.experiments.get(experiment_id)
            if exp is None or exp.user_id != user_id:
                raise NotFoundError(
                    "experiment not found",
                    user_message="Experiment not found.",
                )
            records = uow.models.list_for_experiment(experiment_id)
        else:
            records = uow.models.list_for_user(user_id, limit=limit)
        models = [
            ModelSummaryOut(
                id=r.id,
                name=r.name,
                experiment_id=r.experiment_id,
                task_type=r.task_type,
                family=r.family,
                is_champion=r.is_champion,
                artifact_path=r.artifact_path,
                created_at=r.created_at.isoformat() if r.created_at else None,
            )
            for r in records
        ]
        return ModelsListResponse(models=models, total=len(models))

    def list_experiments(
        self,
        *,
        user_id: str,
        session: Session,
        limit: int = 50,
    ) -> ExperimentsListResponse:
        uow = UnitOfWork(session)
        records = uow.experiments.list_for_user(user_id, limit=limit)
        experiments = [
            ExperimentSummaryOut(
                id=e.id,
                name=e.name,
                status=e.status,
                task_type=e.task_type,
                target_column=e.target_column,
                dataset_id=e.dataset_id,
                created_at=e.created_at.isoformat() if e.created_at else None,
                completed_at=e.completed_at.isoformat() if e.completed_at else None,
            )
            for e in records
        ]
        return ExperimentsListResponse(experiments=experiments, total=len(experiments))

    def list_metrics(
        self,
        *,
        user_id: str,
        session: Session,
        experiment_id: str | None = None,
        model_id: str | None = None,
        limit: int = 500,
    ) -> MetricsListResponse:
        uow = UnitOfWork(session)
        if model_id:
            model = uow.models.get(model_id)
            if model is None or model.user_id != user_id:
                raise NotFoundError("model not found", user_message="Model not found.")
            records = uow.metrics.list_for_model(model_id)[:limit]
        elif experiment_id:
            exp = uow.experiments.get(experiment_id)
            if exp is None or exp.user_id != user_id:
                raise NotFoundError("experiment not found", user_message="Experiment not found.")
            records = uow.metrics.list_for_experiment(experiment_id)[:limit]
        else:
            raise ValidationError(
                "experiment_id or model_id required",
                user_message="Provide experiment_id or model_id to list metrics.",
            )
        metrics = [
            MetricOut(
                id=m.id,
                name=m.name,
                value=float(m.value),
                split=m.split,
                model_id=m.model_id,
                experiment_id=m.experiment_id,
            )
            for m in records
        ]
        return MetricsListResponse(metrics=metrics, total=len(metrics))
