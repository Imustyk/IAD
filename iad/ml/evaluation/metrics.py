"""Classification + regression metrics, computed defensively.

The training service, tuning module and FastAPI backend all share these
functions. Keeping them in a single place ensures the leaderboard, the
hyperparameter search and the production /metrics endpoint always agree.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from iad.core.logging import get_logger

logger = get_logger("iad.ml.evaluation.metrics")


def _round_dict(d: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(float(v), digits) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> dict[str, float]:
    """Standard classification metrics. Works for binary and multiclass."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics["mcc"] = float("nan")

    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                # ``y_proba`` may be shape (n,) or (n, 2). Pick the positive-class column.
                proba_arr = np.asarray(y_proba)
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                    pos_proba = proba_arr[:, 1]
                else:
                    pos_proba = proba_arr.ravel()
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
                metrics["log_loss"] = float(log_loss(y_true, proba_arr, labels=classes))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes)
                )
                metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
        except Exception as exc:  # pragma: no cover — defensive path
            logger.debug("probability-based metrics skipped: %s", exc)
    return _round_dict(metrics)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    metrics = {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": rmse,
    }
    try:
        metrics["mape_%"] = float(
            mean_absolute_percentage_error(
                y_true_arr, y_pred_arr,
            )
            * 100
        )
    except Exception:  # pragma: no cover
        metrics["mape_%"] = float("nan")
    metrics["mean_absolute_residual"] = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    metrics["max_absolute_residual"] = float(np.max(np.abs(y_true_arr - y_pred_arr)))
    return _round_dict(metrics)


# ---------------------------------------------------------------------------
# Naming / scoring helpers — used by the tuning module
# ---------------------------------------------------------------------------
def primary_metric_name(task: Literal["classification", "regression"]) -> str:
    return "f1_macro" if task == "classification" else "r2"


def scoring_for(task: Literal["classification", "regression"]) -> str:
    """sklearn ``scoring`` argument matching :func:`primary_metric_name`."""
    return "f1_macro" if task == "classification" else "r2"
