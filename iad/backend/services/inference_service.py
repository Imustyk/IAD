"""Inference API — load bundles and score rows."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.schemas.ml import PredictionRowOut, PredictRequest, PredictResponse
from iad.backend.services.persistence_service import PersistenceService
from iad.config.settings import get_settings
from iad.core.exceptions import InferenceError, NotFoundError, ValidationError
from iad.core.logging import get_logger
from iad.core.observability.prometheus import observe_ml_operation
from iad.core.validation import validate_inference_payload
from iad.ml.training.persistence import load_bundle

logger = get_logger("iad.backend.inference")


class InferenceService:
    """Application service for ``POST /predict``."""

    def __init__(self, persistence: PersistenceService | None = None) -> None:
        self._persistence = persistence or PersistenceService()

    def _resolve_artifact(
        self,
        request: PredictRequest,
        *,
        session: Session | None,
    ) -> tuple[Path, str | None]:
        if request.artifact_path:
            from iad.core.paths import resolve_trusted_artifact_path

            path = resolve_trusted_artifact_path(request.artifact_path)
            return path, None

        if request.model_id:
            if session is None:
                raise ValidationError(
                    "model_id lookup requires database",
                    user_message="Model registry requires database integration.",
                )
            uow = UnitOfWork(session)
            record = uow.models.get_or_raise(request.model_id)
            if not record.artifact_path:
                raise ValidationError(
                    "Model has no artifact_path",
                    user_message="This model has no saved artifact. Retrain with save_artifact=true.",
                )
            from iad.core.paths import resolve_trusted_artifact_path

            path = resolve_trusted_artifact_path(record.artifact_path)
            return path, record.id

        raise ValidationError(
            "Provide model_id or artifact_path",
            user_message="Specify which model to use via model_id or artifact_path.",
        )

    def predict(
        self,
        request: PredictRequest,
        *,
        session: Session | None = None,
        user_email: str | None = None,
    ) -> PredictResponse:
        settings = get_settings()
        started = time.perf_counter()
        try:
            artifact_path, model_id = self._resolve_artifact(request, session=session)
            pipeline, card = load_bundle(artifact_path)

            frame = pd.DataFrame(request.records)
            aligned = validate_inference_payload(frame, card.features)
            predictions = pipeline.predict(aligned)

            proba_rows: list[dict[str, float] | None] = [None] * len(predictions)
            if request.return_probabilities and card.task == "classification":
                estimator = pipeline.named_steps.get("model")
                if estimator is not None and hasattr(estimator, "predict_proba"):
                    proba = pipeline.predict_proba(aligned)
                    classes = list(estimator.classes_)
                    for i, row in enumerate(proba):
                        proba_rows[i] = {str(classes[j]): float(row[j]) for j in range(len(classes))}

            rows: list[PredictionRowOut] = []
            for i, pred in enumerate(predictions):
                val: Any = pred
                if isinstance(pred, (np.integer, np.floating)):
                    val = pred.item()
                rows.append(PredictionRowOut(prediction=val, probabilities=proba_rows[i]))

            latency_ms = (time.perf_counter() - started) * 1000.0

            if model_id and settings.DATABASE_ENABLED:
                for i, row in enumerate(request.records):
                    self._persistence.log_prediction(
                        model_id=model_id,
                        input_row=row,
                        output=rows[i].prediction,
                        probabilities=rows[i].probabilities,
                        latency_ms=latency_ms / max(len(rows), 1),
                        source="api",
                        user_email=user_email,
                    )

            observe_ml_operation(operation="predict", outcome="success", duration_seconds=latency_ms / 1000.0)
            return PredictResponse(
                model_id=model_id,
                model_name=card.name,
                task_type=card.task,
                predictions=rows,
                latency_ms=round(latency_ms, 2),
            )
        except (InferenceError, ValidationError, NotFoundError):
            observe_ml_operation(operation="predict", outcome="error")
            raise
        except Exception as exc:
            observe_ml_operation(operation="predict", outcome="error")
            logger.exception("predict failed")
            raise InferenceError(
                f"Prediction failed: {exc}",
                user_message="Model inference failed. Check input features match training schema.",
            ) from exc
