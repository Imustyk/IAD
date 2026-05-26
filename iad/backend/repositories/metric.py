"""Metric repository."""
from __future__ import annotations

from sqlalchemy import select

from iad.backend.models.metric import Metric
from iad.backend.repositories.base import BaseRepository


class MetricRepository(BaseRepository[Metric]):
    model = Metric

    def list_for_experiment(self, experiment_id: str) -> list[Metric]:
        stmt = select(Metric).where(Metric.experiment_id == experiment_id)
        return list(self.session.scalars(stmt).all())

    def list_for_model(self, model_id: str) -> list[Metric]:
        stmt = select(Metric).where(Metric.model_id == model_id)
        return list(self.session.scalars(stmt).all())

    def record(
        self,
        *,
        experiment_id: str,
        name: str,
        value: float,
        model_id: str | None = None,
        step: int | None = None,
        split: str | None = None,
    ) -> Metric:
        metric = Metric(
            experiment_id=experiment_id,
            model_id=model_id,
            name=name,
            value=float(value),
            step=step,
            split=split,
        )
        return self.add(metric)

    def record_many(
        self,
        *,
        experiment_id: str,
        metrics: dict[str, float],
        model_id: str | None = None,
        split: str | None = None,
    ) -> list[Metric]:
        return [
            self.record(
                experiment_id=experiment_id,
                model_id=model_id,
                name=k,
                value=v,
                split=split,
            )
            for k, v in metrics.items()
        ]
