"""Inference audit log — inputs, outputs, latency."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from iad.backend.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from iad.backend.models.ml_model import MLModelRecord
    from iad.backend.models.user import User


class Prediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "predictions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    probability_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)  # api | streamlit

    owner: Mapped[User] = relationship(back_populates="predictions")
    model: Mapped[MLModelRecord] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_model_created", "model_id", "created_at"),
        Index("ix_predictions_user_created", "user_id", "created_at"),
    )
