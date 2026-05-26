"""Experiment and model evaluation metrics."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from iad.backend.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from iad.backend.models.experiment import Experiment
    from iad.backend.models.ml_model import MLModelRecord


class Metric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "metrics"

    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    split: Mapped[str | None] = mapped_column(String(32), nullable=True)  # train | test | cv

    experiment: Mapped[Experiment] = relationship(back_populates="metrics")
    model: Mapped[MLModelRecord | None] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("ix_metrics_experiment_name", "experiment_id", "name"),
        Index("ix_metrics_model_name", "model_id", "name"),
    )
