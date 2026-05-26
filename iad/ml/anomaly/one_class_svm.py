"""One-Class SVM anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.anomaly.reports import AnomalyReport

logger = get_logger("iad.ml.anomaly.one_class_svm")


def detect_one_class_svm(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> AnomalyReport:
    """Fit One-Class SVM on inlier-heavy numeric data."""
    numeric = df.select_dtypes(include=["number"])
    cols = list(feature_columns) if feature_columns is not None else numeric.columns.tolist()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Columns not found: {missing}",
            user_message="Select valid numeric columns.",
        )
    if not cols:
        raise SchemaError(
            "No numeric columns for anomaly detection.",
            user_message="Load data with numeric features.",
        )

    matrix = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matrix) < 10:
        raise AnalyticsError(
            "Need at least 10 rows for One-Class SVM.",
            user_message="Add more observations.",
        )

    nu = float(np.clip(nu, 0.01, 0.5))
    scaled = StandardScaler().fit_transform(matrix)
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    preds = model.fit_predict(scaled)
    scores = -model.decision_function(scaled).ravel()
    threshold = float(np.quantile(scores, 1 - nu))
    is_anomaly = pd.Series(preds == -1, index=matrix.index, name="is_anomaly")

    logger.info(
        "one-class svm anomalies",
        extra={"n_anomalies": int(is_anomaly.sum()), "nu": nu},
    )
    return AnomalyReport(
        method="one_class_svm",
        scores=pd.Series(scores, index=matrix.index, name="anomaly_score"),
        is_anomaly=is_anomaly,
        threshold=threshold,
        feature_columns=cols,
        metrics={
            "anomaly_count": float(is_anomaly.sum()),
            "anomaly_share": float(is_anomaly.mean()),
            "nu": nu,
        },
    )
