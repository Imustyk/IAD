"""Isolation Forest multivariate anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.anomaly.reports import AnomalyReport

logger = get_logger("iad.ml.anomaly.isolation_forest")


def detect_isolation_forest(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyReport:
    """Flag anomalies using IsolationForest on numeric columns."""
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
            "Need at least 10 rows for Isolation Forest.",
            user_message="Add more observations.",
        )

    contamination = float(np.clip(contamination, 0.01, 0.5))
    scaled = StandardScaler().fit_transform(matrix)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = model.fit_predict(scaled)
    scores = -model.score_samples(scaled)
    threshold = float(np.quantile(scores, 1 - contamination))
    is_anomaly = pd.Series(preds == -1, index=matrix.index, name="is_anomaly")

    logger.info(
        "isolation forest anomalies",
        extra={"n_anomalies": int(is_anomaly.sum()), "contamination": contamination},
    )
    return AnomalyReport(
        method="isolation_forest",
        scores=pd.Series(scores, index=matrix.index, name="anomaly_score"),
        is_anomaly=is_anomaly,
        threshold=threshold,
        feature_columns=cols,
        metrics={
            "anomaly_count": float(is_anomaly.sum()),
            "anomaly_share": float(is_anomaly.mean()),
            "contamination": contamination,
        },
    )
