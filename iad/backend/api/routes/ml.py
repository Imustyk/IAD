"""ML platform routes — train, predict, models, experiments, metrics, upload."""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from iad.backend.api.deps import DbSession, require_perm
from iad.backend.schemas.ml import (
    DatasetUploadResponse,
    ExperimentsListResponse,
    MetricsListResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from iad.backend.security.permissions import Permission
from iad.backend.services.auth_service import AuthenticatedUser
from iad.backend.services.dataset_upload_service import DatasetUploadService
from iad.backend.services.inference_service import InferenceService
from iad.backend.services.ml_catalog_service import MLCatalogService
from iad.backend.services.training_api_service import TrainingAPIService
from iad.core.exceptions import IADError, UploadError, ValidationError

router = APIRouter(tags=["ml"])

TrainUser = Annotated[AuthenticatedUser, Depends(require_perm(Permission.MODEL_TRAIN))]
PredictUser = Annotated[AuthenticatedUser, Depends(require_perm(Permission.MODEL_PREDICT))]
DatasetUser = Annotated[AuthenticatedUser, Depends(require_perm(Permission.DATASET_WRITE))]
CatalogUser = Annotated[AuthenticatedUser, Depends(require_perm(Permission.EXPERIMENT_READ))]


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
async def train_model(body: TrainRequest, user: TrainUser, db: DbSession) -> TrainResponse:
    """Train a leaderboard of models on inline data or a registered dataset."""
    try:
        return await asyncio.to_thread(
            TrainingAPIService().train,
            body,
            session=db,
            user_email=user.email,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    except IADError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.user_message) from exc


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, user: PredictUser, db: DbSession) -> PredictResponse:
    """Score one or more rows with a saved model bundle."""
    try:
        return await asyncio.to_thread(
            InferenceService().predict,
            body,
            session=db,
            user_email=user.email,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    except IADError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.user_message) from exc


@router.get("/models", response_model=ModelsListResponse)
def list_models(
    user: CatalogUser,
    db: DbSession,
    experiment_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> ModelsListResponse:
    """List trained models for the authenticated user."""
    return MLCatalogService().list_models(
        user_id=user.id,
        session=db,
        experiment_id=experiment_id,
        limit=limit,
    )


@router.get("/experiments", response_model=ExperimentsListResponse)
def list_experiments(
    user: CatalogUser,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
) -> ExperimentsListResponse:
    """List ML experiment runs."""
    return MLCatalogService().list_experiments(user_id=user.id, session=db, limit=limit)


@router.get("/ml/metrics", response_model=MetricsListResponse)
def list_ml_metrics(
    user: CatalogUser,
    db: DbSession,
    experiment_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
) -> MetricsListResponse:
    """List stored evaluation metrics (Prometheus scrape remains at ``GET /metrics``)."""
    try:
        return MLCatalogService().list_metrics(
            user_id=user.id,
            session=db,
            experiment_id=experiment_id,
            model_id=model_id,
            limit=limit,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    user: DatasetUser,
    db: DbSession,
    file: UploadFile = File(...),
    name: str | None = Query(default=None),
    description: str | None = Query(default=None),
) -> DatasetUploadResponse:
    """Upload a tabular dataset (CSV, Parquet, Excel, JSON)."""
    filename = file.filename or "upload.csv"
    try:
        data = await file.read()
        _df, response = await asyncio.to_thread(
            DatasetUploadService().upload,
            filename=filename,
            data=data,
            content_type=file.content_type,
            name=name,
            description=description,
            user_email=user.email,
            session=db,
        )
        return response
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    except IADError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.user_message) from exc
