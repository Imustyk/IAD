"""Clustering analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ClusteringReport:
    """Labels, metrics, and 2-D projection for visualization."""

    method: str
    labels: pd.Series
    n_clusters: int
    metrics: dict[str, float] = field(default_factory=dict)
    projection: pd.DataFrame | None = None
    projection_method: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    cluster_sizes: pd.DataFrame | None = None

    def labeled_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["cluster"] = self.labels.values
        return out
