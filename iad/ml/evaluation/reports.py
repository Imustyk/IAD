"""Structured evaluation reports — confusion matrix, regression diagnostics, calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ConfusionMatrixReport:
    """Confusion matrix + per-class precision / recall / support."""

    labels: list[str]
    matrix: pd.DataFrame
    per_class: pd.DataFrame
    accuracy: float
    n_samples: int


def build_confusion_matrix_report(
    y_true, y_pred, labels: list | None = None
) -> ConfusionMatrixReport:
    if labels is None:
        labels = sorted(pd.Series(y_true).dropna().unique().tolist(), key=str)
    label_strs = [str(label) for label in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    matrix = pd.DataFrame(cm, index=label_strs, columns=label_strs)

    per_class_rows = []
    total = matrix.values.sum()
    for i, label in enumerate(label_strs):
        row_sum = matrix.iloc[i, :].sum()
        col_sum = matrix.iloc[:, i].sum()
        tp = matrix.iloc[i, i]
        precision = tp / col_sum if col_sum else 0.0
        recall = tp / row_sum if row_sum else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_rows.append(
            {
                "class": label,
                "support": int(row_sum),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    accuracy = float(np.trace(cm) / total) if total else 0.0
    return ConfusionMatrixReport(
        labels=label_strs,
        matrix=matrix,
        per_class=per_class,
        accuracy=round(accuracy, 4),
        n_samples=int(total),
    )


@dataclass(frozen=True)
class RegressionReport:
    """Predicted vs actual + residuals, ready for plotting."""

    predictions: pd.DataFrame  # columns: actual, predicted, residual
    residual_quantiles: dict[str, float]
    rmse: float
    mae: float
    r2: float


def build_regression_report(y_true, y_pred) -> RegressionReport:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    df = pd.DataFrame(
        {"actual": y_true_arr, "predicted": y_pred_arr, "residual": residuals}
    )
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(residuals, q))
        for q in (0.05, 0.25, 0.50, 0.75, 0.95)
    }
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return RegressionReport(
        predictions=df,
        residual_quantiles=quantiles,
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(float(r2), 4),
    )


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability data for binary classifiers (expected vs observed probabilities)."""

    fraction_of_positives: list[float]
    mean_predicted_value: list[float]
    n_bins: int
    brier_score: float = field(default=float("nan"))


def build_calibration_report(
    y_true, y_proba, n_bins: int = 10
) -> CalibrationReport:
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, -1]
    fraction, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    brier = float(np.mean((proba - np.asarray(y_true, dtype=float)) ** 2))
    return CalibrationReport(
        fraction_of_positives=[round(float(v), 4) for v in fraction],
        mean_predicted_value=[round(float(v), 4) for v in mean_pred],
        n_bins=n_bins,
        brier_score=round(brier, 4),
    )
