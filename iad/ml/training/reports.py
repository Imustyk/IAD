"""Structured report dataclasses returned by the TrainingService."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from iad.ml.evaluation.reports import (
    CalibrationReport,
    ConfusionMatrixReport,
    RegressionReport,
)
from iad.ml.training.reproducibility import ModelCard


@dataclass(frozen=True)
class LeaderboardEntry:
    """A single row of the leaderboard."""

    model_name: str
    family: str
    metrics: dict[str, float]
    cv_metrics: dict[str, float] = field(default_factory=dict)
    train_time_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Everything produced by a single ``TrainingService.train`` invocation."""

    task: str
    target: str
    features: list[str]
    schema_groups: dict[str, list[str]]
    leaderboard: list[LeaderboardEntry]
    best_pipeline: Pipeline
    best_entry: LeaderboardEntry
    test_predictions: pd.DataFrame
    confusion_matrix: ConfusionMatrixReport | None = None
    regression_report: RegressionReport | None = None
    calibration: CalibrationReport | None = None
    feature_importance: pd.DataFrame | None = None
    model_card: ModelCard | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def leaderboard_frame(self) -> pd.DataFrame:
        rows = []
        for entry in self.leaderboard:
            row = {"model": entry.model_name, "family": entry.family}
            row.update(entry.metrics)
            row.update(entry.cv_metrics)
            row["train_time_s"] = round(entry.train_time_seconds, 3)
            if entry.error:
                row["error"] = entry.error
            rows.append(row)
        return pd.DataFrame(rows)
