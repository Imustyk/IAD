"""Dataset repository."""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import desc, select

from iad.backend.models.dataset import Dataset
from iad.backend.repositories.base import BaseRepository

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower().strip())
    return slug.strip("-")[:200] or "dataset"


class DatasetRepository(BaseRepository[Dataset]):
    model = Dataset

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[Dataset]:
        stmt = (
            select(Dataset)
            .where(Dataset.user_id == user_id, Dataset.is_active.is_(True))
            .order_by(desc(Dataset.created_at))
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def get_latest_version(self, user_id: str, slug: str) -> Dataset | None:
        stmt = (
            select(Dataset)
            .where(Dataset.user_id == user_id, Dataset.slug == slug)
            .order_by(desc(Dataset.version))
            .limit(1)
        )
        return self.session.scalar(stmt)

    def create_version(
        self,
        *,
        user_id: str,
        name: str,
        row_count: int | None = None,
        column_count: int | None = None,
        schema_json: dict[str, Any] | None = None,
        storage_path: str | None = None,
        checksum_sha256: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> Dataset:
        slug = slugify(name)
        latest = self.get_latest_version(user_id, slug)
        version = (latest.version + 1) if latest else 1
        record = Dataset(
            user_id=user_id,
            name=name,
            slug=slug,
            version=version,
            row_count=row_count,
            column_count=column_count,
            schema_json=schema_json,
            storage_path=storage_path,
            checksum_sha256=checksum_sha256,
            source=source,
            description=description,
        )
        return self.add(record)
