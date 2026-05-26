"""Dataset upload use case — validate, persist file, register metadata."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.schemas.ml import DatasetUploadResponse
from iad.backend.security.upload_policy import validate_upload
from iad.backend.services.dataframe_io import load_dataframe_from_bytes
from iad.backend.services.persistence_service import PersistenceService
from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.core.paths import safe_filename

logger = get_logger("iad.backend.dataset_upload")


class DatasetUploadService:
    """Handle ``POST /upload`` — store file + optional DB registration."""

    def __init__(self, persistence: PersistenceService | None = None) -> None:
        self._persistence = persistence or PersistenceService()

    def upload(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str | None,
        name: str | None = None,
        description: str | None = None,
        user_email: str | None = None,
        session: Session | None = None,
    ) -> tuple[pd.DataFrame, DatasetUploadResponse]:
        settings = get_settings()
        validate_upload(filename=filename, size_bytes=len(data), content_type=content_type)

        df = load_dataframe_from_bytes(data, filename)
        display_name = name or Path(filename).stem
        upload_dir = settings.DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex[:12]}_{safe_filename(filename)}"
        storage_path = upload_dir / safe_name
        storage_path.write_bytes(data)

        dataset_id = ""
        slug = ""
        version = 1
        if session is not None:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create_default(user_email or PersistenceService.DEFAULT_USER_EMAIL)
            record = uow.datasets.create_version(
                user_id=user.id,
                name=display_name,
                row_count=int(df.shape[0]),
                column_count=int(df.shape[1]),
                schema_json=PersistenceService._schema_from_dataframe(df),
                storage_path=str(storage_path),
                checksum_sha256=PersistenceService._checksum(df),
                source="api_upload",
                description=description,
            )
            dataset_id = record.id
            slug = record.slug
            version = record.version
        else:
            result = self._persistence.register_dataset(
                df,
                name=display_name,
                storage_path=storage_path,
                source="api_upload",
                description=description,
                user_email=user_email,
            )
            dataset_id = result.dataset_id
            slug = result.slug
            version = result.version

        response = DatasetUploadResponse(
            dataset_id=dataset_id,
            name=display_name,
            slug=slug,
            version=version,
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            storage_path=str(storage_path),
            schema_columns=[str(c) for c in df.columns],
        )
        logger.info("dataset uploaded", extra={"dataset_id": dataset_id, "ctx_path": str(storage_path)})
        return df, response
