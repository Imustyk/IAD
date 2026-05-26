"""Anomaly detection service."""
from __future__ import annotations

import pandas as pd

from iad.ml.anomaly.isolation_forest import detect_isolation_forest
from iad.ml.anomaly.one_class_svm import detect_one_class_svm
from iad.ml.anomaly.reports import AnomalyReport


class AnomalyService:
    """Multivariate anomaly detection facade."""

    def isolation_forest(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> AnomalyReport:
        return detect_isolation_forest(
            df,
            feature_columns=feature_columns,
            contamination=contamination,
            random_state=random_state,
        )

    def one_class_svm(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        nu: float = 0.05,
    ) -> AnomalyReport:
        return detect_one_class_svm(df, feature_columns=feature_columns, nu=nu)
