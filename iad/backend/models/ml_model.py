"""Trained model registry records (artifact paths + metadata)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from iad.backend.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from iad.backend.models.experiment import Experiment
    from iad.backend.models.metric import Metric
    from iad.backend.models.prediction import Prediction
    from iad.backend.models.user import User


class MLModelRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Persisted trained model — table name ``ml_models`` avoids ORM ``Model`` clash."""

    __tablename__ = "ml_models"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    family: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    model_card_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    is_champion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    version_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    owner: Mapped[User] = relationship(back_populates="models")
    experiment: Mapped[Experiment] = relationship(back_populates="models")
    metrics: Mapped[list[Metric]] = relationship(back_populates="model", cascade="all, delete-orphan")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_ml_models_experiment_champion", "experiment_id", "is_champion"),
        Index("ix_ml_models_user_created", "user_id", "created_at"),
    )
