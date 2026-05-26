"""Generic repository with typed CRUD operations."""
from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from iad.backend.database.base import Base
from iad.core.exceptions import NotFoundError

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Data-access object for a single ORM model."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity_id: str) -> ModelT | None:
        return self.session.get(self.model, entity_id)

    def get_or_raise(self, entity_id: str) -> ModelT:
        entity = self.get(entity_id)
        if entity is None:
            raise NotFoundError(
                f"{self.model.__name__} {entity_id} not found",
                user_message="The requested item was not found.",
                entity=self.model.__name__,
                entity_id=entity_id,
            )
        return entity

    def add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        self.session.flush()
        return entity

    def delete(self, entity: ModelT) -> None:
        self.session.delete(entity)
        self.session.flush()

    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        return list(self.session.scalars(stmt).all())
