"""Prediction audit repository."""
from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select

from iad.backend.models.prediction import Prediction
from iad.backend.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    def list_for_model(self, model_id: str, *, limit: int = 100) -> list[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.model_id == model_id)
            .order_by(desc(Prediction.created_at))
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def log(
        self,
        *,
        user_id: str,
        model_id: str,
        input_json: dict[str, Any],
        output_json: dict[str, Any],
        probability_json: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        batch_size: int = 1,
        source: str | None = None,
    ) -> Prediction:
        record = Prediction(
            user_id=user_id,
            model_id=model_id,
            input_json=input_json,
            output_json=output_json,
            probability_json=probability_json,
            latency_ms=latency_ms,
            batch_size=batch_size,
            source=source,
        )
        return self.add(record)
