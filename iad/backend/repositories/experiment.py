"""Experiment repository."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select

from iad.backend.models.experiment import Experiment, ExperimentStatus
from iad.backend.repositories.base import BaseRepository


class ExperimentRepository(BaseRepository[Experiment]):
    model = Experiment

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[Experiment]:
        stmt = (
            select(Experiment)
            .where(Experiment.user_id == user_id)
            .order_by(desc(Experiment.created_at))
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def create(
        self,
        *,
        user_id: str,
        name: str,
        task_type: str,
        target_column: str,
        feature_columns: list[str],
        dataset_id: str | None = None,
        config_json: dict[str, Any] | None = None,
    ) -> Experiment:
        exp = Experiment(
            user_id=user_id,
            dataset_id=dataset_id,
            name=name,
            task_type=task_type,
            target_column=target_column,
            feature_columns=feature_columns,
            config_json=config_json,
            status=ExperimentStatus.PENDING.value,
        )
        return self.add(exp)

    def mark_running(self, experiment_id: str) -> Experiment:
        exp = self.get_or_raise(experiment_id)
        exp.status = ExperimentStatus.RUNNING.value
        exp.started_at = datetime.now(UTC)
        self.session.flush()
        return exp

    def mark_completed(self, experiment_id: str) -> Experiment:
        exp = self.get_or_raise(experiment_id)
        exp.status = ExperimentStatus.COMPLETED.value
        exp.completed_at = datetime.now(UTC)
        self.session.flush()
        return exp

    def mark_failed(self, experiment_id: str, error_message: str) -> Experiment:
        exp = self.get_or_raise(experiment_id)
        exp.status = ExperimentStatus.FAILED.value
        exp.error_message = error_message[:4000]
        exp.completed_at = datetime.now(UTC)
        self.session.flush()
        return exp
