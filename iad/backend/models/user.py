"""User ORM model — authentication fields reserved for Phase 7."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from iad.backend.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from iad.backend.models.dataset import Dataset
    from iad.backend.models.experiment import Experiment
    from iad.backend.models.ml_model import MLModelRecord
    from iad.backend.models.prediction import Prediction


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    is_superuser: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="analyst", server_default="analyst"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    models: Mapped[list[MLModelRecord]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_users_active_email", "is_active", "email"),
    )
