"""Training bridge — legacy ``src.predictive`` and Phase 3 ``TrainingService``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from iad.core.logging import get_logger
from iad.ml.training.registry import ModelRegistry
from iad.ml.training.reports import TrainingResult
from iad.ml.training.service import TrainingConfig, TrainingService

logger = get_logger("iad.frontend.training")

TaskType = Literal["classification", "regression"]


@dataclass
class UnifiedTrainingReport:
    """UI-friendly report compatible with legacy page rendering."""

    task_type: str
    target: str
    features: list[str]
    leaderboard: pd.DataFrame
    best_model_name: str
    metrics: dict[str, float]
    cv_metrics: dict[str, float]
    confusion_matrix: pd.DataFrame | None = None
    classes: list[str] | None = None
    feature_importance: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    test_predictions: pd.DataFrame | None = None
    training_result: TrainingResult | None = None
    engine: str = "legacy"


def _result_to_unified(result: TrainingResult) -> UnifiedTrainingReport:
    best = result.best_entry
    metrics = dict(best.metrics)
    cv_metrics = dict(best.cv_metrics)

    cm_df = None
    if result.confusion_matrix is not None:
        cm_df = result.confusion_matrix.matrix

    residuals_df = None
    if result.regression_report is not None:
        residuals_df = result.regression_report.predictions

    return UnifiedTrainingReport(
        task_type=result.task,
        target=result.target,
        features=list(result.features),
        leaderboard=result.leaderboard_frame(),
        best_model_name=best.model_name,
        metrics=metrics,
        cv_metrics=cv_metrics,
        confusion_matrix=cm_df,
        feature_importance=result.feature_importance,
        residuals=residuals_df,
        test_predictions=result.test_predictions,
        training_result=result,
        engine="enterprise",
    )


def train_legacy(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    task_type: TaskType,
    test_size: float,
    random_state: int,
    cv_folds: int,
    selected_models: list[str] | None,
) -> tuple[Pipeline, UnifiedTrainingReport]:
    from src.predictive import train_models

    pipeline, report = train_models(
        df=df,
        target=target,
        feature_columns=feature_columns,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        selected_models=selected_models,
    )
    unified = UnifiedTrainingReport(
        task_type=report.task_type,
        target=report.target,
        features=report.features,
        leaderboard=report.leaderboard,
        best_model_name=report.best_model_name,
        metrics=report.metrics,
        cv_metrics=report.cv_metrics,
        confusion_matrix=report.confusion_matrix,
        classes=report.classes,
        feature_importance=report.feature_importance,
        residuals=report.residuals,
        test_predictions=report.test_predictions,
        engine="legacy",
    )
    return pipeline, unified


def train_enterprise(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    task_type: TaskType,
    test_size: float,
    random_state: int,
    cv_folds: int,
    selected_models: list[str] | None,
    track_mlflow: bool = False,
) -> tuple[Pipeline, UnifiedTrainingReport]:
    service = TrainingService(ModelRegistry.default())
    config = TrainingConfig(
        task=task_type,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state,
        selected_models=selected_models,
        track_mlflow=track_mlflow,
    )
    result = service.train(
        df,
        target=target,
        config=config,
        feature_columns=feature_columns,
    )
    unified = _result_to_unified(result)
    return result.best_pipeline, unified


def get_training_service() -> TrainingService:
    """Return a shared TrainingService (not cached — safe for tests)."""
    return TrainingService(ModelRegistry.default())
