"""Multivariate anomaly detection results."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class AnomalyReport:
    """Row-level anomaly flags and scores."""

    method: str
    scores: pd.Series
    is_anomaly: pd.Series
    threshold: float
    feature_columns: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def flagged_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[self.scores.index].copy()
        out["anomaly_score"] = self.scores
        out["is_anomaly"] = self.is_anomaly
        return out
