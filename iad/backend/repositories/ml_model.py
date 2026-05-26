"""Trained model repository."""
from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select, update

from iad.backend.models.ml_model import MLModelRecord
from iad.backend.repositories.base import BaseRepository


class MLModelRepository(BaseRepository[MLModelRecord]):
    model = MLModelRecord

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[MLModelRecord]:
        stmt = (
            select(MLModelRecord)
            .where(MLModelRecord.user_id == user_id)
            .order_by(desc(MLModelRecord.created_at))
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def list_for_experiment(self, experiment_id: str) -> list[MLModelRecord]:
        stmt = (
            select(MLModelRecord)
            .where(MLModelRecord.experiment_id == experiment_id)
            .order_by(desc(MLModelRecord.created_at))
        )
        return list(self.session.scalars(stmt).all())

    def get_champion(self, experiment_id: str) -> MLModelRecord | None:
        stmt = select(MLModelRecord).where(
            MLModelRecord.experiment_id == experiment_id,
            MLModelRecord.is_champion.is_(True),
        )
        return self.session.scalar(stmt)

    def create(
        self,
        *,
        user_id: str,
        experiment_id: str,
        name: str,
        task_type: str,
        family: str | None = None,
        artifact_path: str | None = None,
        model_card_json: dict[str, Any] | None = None,
        is_champion: bool = False,
    ) -> MLModelRecord:
        if is_champion:
            self._clear_champion(experiment_id)
        record = MLModelRecord(
            user_id=user_id,
            experiment_id=experiment_id,
            name=name,
            task_type=task_type,
            family=family,
            artifact_path=artifact_path,
            model_card_json=model_card_json,
            is_champion=is_champion,
        )
        return self.add(record)

    def set_champion(self, model_id: str) -> MLModelRecord:
        record = self.get_or_raise(model_id)
        self._clear_champion(record.experiment_id)
        record.is_champion = True
        self.session.flush()
        return record

    def _clear_champion(self, experiment_id: str) -> None:
        stmt = (
            update(MLModelRecord)
            .where(MLModelRecord.experiment_id == experiment_id)
            .values(is_champion=False)
        )
        self.session.execute(stmt)
