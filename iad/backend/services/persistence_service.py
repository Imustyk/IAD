"""High-level persistence orchestration for datasets, experiments, and models.

Consumed by:
    * FastAPI routes (Phase 5)
    * Optional Streamlit hooks when ``DATABASE_ENABLED=true``

Design:
    * One :class:`UnitOfWork` per public method — explicit transaction boundaries.
    * DataFrame metadata extracted without storing full frames in PostgreSQL
      (large payloads go to ``storage_path`` / object storage in Phase 5+).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from iad.backend.database.session import session_scope
from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.ml.training.reports import TrainingResult

logger = get_logger("iad.persistence")


@dataclass(frozen=True)
class DatasetRecordResult:
    dataset_id: str
    user_id: str
    version: int
    slug: str


@dataclass(frozen=True)
class ExperimentPersistResult:
    experiment_id: str
    champion_model_id: str | None
    metric_count: int


class PersistenceService:
    """Application service for relational persistence."""

    DEFAULT_USER_EMAIL = "default@iad.local"

    def __init__(self, *, default_user_email: str | None = None) -> None:
        self.default_user_email = default_user_email or self.DEFAULT_USER_EMAIL

    def _default_user_id(self, uow: UnitOfWork) -> str:
        user = uow.users.get_or_create_default(self.default_user_email)
        return user.id

    @staticmethod
    def _schema_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "columns": [
                {
                    "name": str(c),
                    "dtype": str(df[c].dtype),
                    "non_null": int(df[c].notna().sum()),
                    "unique": int(df[c].nunique(dropna=True)),
                }
                for c in df.columns
            ],
            "shape": list(df.shape),
        }

    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(df.head(500), index=True).values.tobytes())
        h.update(str(df.shape).encode())
        return h.hexdigest()

    def register_dataset(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        source: str | None = None,
        storage_path: str | Path | None = None,
        description: str | None = None,
        user_email: str | None = None,
    ) -> DatasetRecordResult:
        """Persist dataset metadata (and optional file path)."""
        with session_scope() as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create_default(user_email or self.default_user_email)
            record = uow.datasets.create_version(
                user_id=user.id,
                name=name,
                row_count=int(df.shape[0]),
                column_count=int(df.shape[1]),
                schema_json=self._schema_from_dataframe(df),
                storage_path=str(storage_path) if storage_path else None,
                checksum_sha256=self._checksum(df),
                source=source,
                description=description,
            )
            logger.info(
                "dataset registered",
                extra={
                    "dataset_id": record.id,
                    "dataset_slug": record.slug,
                    "dataset_version": record.version,
                },
            )
            return DatasetRecordResult(
                dataset_id=record.id,
                user_id=user.id,
                version=record.version,
                slug=record.slug,
            )

    def persist_training_result(
        self,
        result: TrainingResult,
        *,
        experiment_name: str,
        dataset_id: str | None = None,
        artifact_path: str | Path | None = None,
        user_email: str | None = None,
    ) -> ExperimentPersistResult:
        """Store experiment, champion model, and leaderboard metrics."""
        with session_scope() as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create_default(user_email or self.default_user_email)

            exp = uow.experiments.create(
                user_id=user.id,
                dataset_id=dataset_id,
                name=experiment_name,
                task_type=result.task,
                target_column=result.target,
                feature_columns=list(result.features),
                config_json=result.extra,
            )
            uow.experiments.mark_running(exp.id)

            champion_id: str | None = None
            metric_count = 0

            for entry in result.leaderboard:
                if entry.error:
                    continue
                model = uow.models.create(
                    user_id=user.id,
                    experiment_id=exp.id,
                    name=entry.model_name,
                    task_type=result.task,
                    family=entry.family,
                    is_champion=(entry.model_name == result.best_entry.model_name),
                )
                if entry.model_name == result.best_entry.model_name:
                    champion_id = model.id
                    if artifact_path:
                        model.artifact_path = str(artifact_path)
                    if result.model_card is not None:
                        model.model_card_json = result.model_card.to_dict()

                uow.metrics.record_many(
                    experiment_id=exp.id,
                    model_id=model.id,
                    metrics=entry.metrics,
                    split="test",
                )
                metric_count += len(entry.metrics)
                if entry.cv_metrics:
                    uow.metrics.record_many(
                        experiment_id=exp.id,
                        model_id=model.id,
                        metrics=entry.cv_metrics,
                        split="cv",
                    )
                    metric_count += len(entry.cv_metrics)

            if champion_id:
                uow.models.set_champion(champion_id)

            uow.experiments.mark_completed(exp.id)
            logger.info(
                "experiment persisted",
                extra={"experiment_id": exp.id, "champion_model_id": champion_id},
            )
            return ExperimentPersistResult(
                experiment_id=exp.id,
                champion_model_id=champion_id,
                metric_count=metric_count,
            )

    def log_prediction(
        self,
        *,
        model_id: str,
        input_row: dict[str, Any],
        output: Any,
        probabilities: dict[str, float] | None = None,
        latency_ms: float | None = None,
        source: str = "streamlit",
        user_email: str | None = None,
    ) -> str:
        """Append an inference audit row."""
        with session_scope() as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create_default(user_email or self.default_user_email)
            record = uow.predictions.log(
                user_id=user.id,
                model_id=model_id,
                input_json=input_row,
                output_json={"prediction": output},
                probability_json=probabilities,
                latency_ms=latency_ms,
                source=source,
            )
            return record.id

    def health(self) -> dict[str, Any]:
        from iad.backend.database.session import check_connection

        settings = get_settings()
        return {
            "enabled": settings.DATABASE_ENABLED,
            "url_scheme": settings.resolved_database_url().split("://")[0],
            "connected": check_connection(settings),
        }
