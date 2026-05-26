"""Evaluation metrics + reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

from iad.ml.evaluation import (
    classification_metrics,
    primary_metric_name,
    regression_metrics,
    scoring_for,
)
from iad.ml.evaluation.reports import (
    build_calibration_report,
    build_confusion_matrix_report,
    build_regression_report,
)


def test_classification_metrics_binary() -> None:
    y_true = [0, 1, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 0, 1]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    metrics = classification_metrics(y_true, y_pred, y_proba=proba)
    assert 0 <= metrics["accuracy"] <= 1
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics
    assert "log_loss" in metrics


def test_classification_metrics_multiclass() -> None:
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 2, 0, 0, 2]
    metrics = classification_metrics(y_true, y_pred)
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "mcc" in metrics


def test_regression_metrics_basic() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0, 51.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["r2"] > 0.9
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_primary_metric_and_scoring() -> None:
    assert primary_metric_name("classification") == "f1_macro"
    assert primary_metric_name("regression") == "r2"
    assert scoring_for("classification") == "f1_macro"
    assert scoring_for("regression") == "r2"


def test_confusion_matrix_report() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    rep = build_confusion_matrix_report(y_true, y_pred)
    assert isinstance(rep.matrix, pd.DataFrame)
    assert rep.n_samples == 6
    assert rep.matrix.shape == (3, 3)
    assert "precision" in rep.per_class.columns


def test_regression_report_residuals() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.0])
    rep = build_regression_report(y_true, y_pred)
    assert rep.predictions.shape == (4, 3)
    assert rep.r2 > 0.9
    assert "q50" in rep.residual_quantiles


def test_calibration_report() -> None:
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=200)
    y = (proba > 0.5).astype(int)
    rep = build_calibration_report(y, proba, n_bins=5)
    assert rep.n_bins == 5
    assert len(rep.fraction_of_positives) <= 5
