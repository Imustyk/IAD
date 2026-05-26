"""Pydantic schemas for ML API — train, predict, catalog, upload."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TrainRequest(BaseModel):
    """Request body for ``POST /train``."""

    target_column: str = Field(..., min_length=1, max_length=255)
    task_type: Literal["classification", "regression"] | None = None
    experiment_name: str = Field(default="api_training", max_length=255)
    dataset_id: str | None = Field(default=None, description="Use a previously uploaded dataset")
    records: list[dict[str, Any]] | None = Field(
        default=None,
        description="Inline tabular rows (alternative to dataset_id)",
    )
    feature_columns: list[str] | None = None
    selected_models: list[str] | None = Field(
        default=None,
        description="Subset of registry model names; null trains all candidates",
    )
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=20)
    cross_validate_best: bool = True
    track_mlflow: bool = False
    mlflow_experiment: str = "iad_api"
    persist_to_database: bool = True
    save_artifact: bool = True
    notes: str = ""

    @field_validator("records")
    @classmethod
    def _validate_records(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is not None and len(v) < 2:
            raise ValueError("records must contain at least 2 rows for training")
        return v


class LeaderboardRowOut(BaseModel):
    model_name: str
    family: str
    metrics: dict[str, float]
    cv_metrics: dict[str, float] = Field(default_factory=dict)
    train_time_seconds: float = 0.0
    error: str | None = None


class TrainResponse(BaseModel):
    experiment_id: str | None = None
    champion_model_id: str | None = None
    artifact_path: str | None = None
    task_type: str
    target_column: str
    best_model_name: str
    metrics: dict[str, float]
    cv_metrics: dict[str, float] = Field(default_factory=dict)
    leaderboard: list[LeaderboardRowOut]
    features: list[str]


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    records: list[dict[str, Any]] = Field(..., min_length=1)
    model_id: str | None = Field(default=None, description="Registered model UUID")
    artifact_path: str | None = Field(
        default=None,
        description="Direct path to .joblib bundle (bypasses registry)",
    )
    return_probabilities: bool = False

    @field_validator("records")
    @classmethod
    def _non_empty_records(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("records must not be empty")
        return v


class PredictionRowOut(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    model_id: str | None = None
    model_name: str
    task_type: str
    predictions: list[PredictionRowOut]
    latency_ms: float


class ModelSummaryOut(BaseModel):
    id: str
    name: str
    experiment_id: str
    task_type: str
    family: str | None = None
    is_champion: bool
    artifact_path: str | None = None
    created_at: str | None = None


class ModelsListResponse(BaseModel):
    models: list[ModelSummaryOut]
    total: int


class ExperimentSummaryOut(BaseModel):
    id: str
    name: str
    status: str
    task_type: str
    target_column: str
    dataset_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class ExperimentsListResponse(BaseModel):
    experiments: list[ExperimentSummaryOut]
    total: int


class MetricOut(BaseModel):
    id: str
    name: str
    value: float
    split: str | None = None
    model_id: str | None = None
    experiment_id: str


class MetricsListResponse(BaseModel):
    metrics: list[MetricOut]
    total: int


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    slug: str
    version: int
    rows: int
    columns: int
    storage_path: str
    schema_columns: list[str]
