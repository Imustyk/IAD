"""Model evaluation: metrics + structured reports."""
from iad.ml.evaluation.metrics import (
    classification_metrics,
    primary_metric_name,
    regression_metrics,
    scoring_for,
)
from iad.ml.evaluation.reports import (
    CalibrationReport,
    ConfusionMatrixReport,
    RegressionReport,
)

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "primary_metric_name",
    "scoring_for",
    "ConfusionMatrixReport",
    "RegressionReport",
    "CalibrationReport",
]
