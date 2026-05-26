"""Clustering service."""
from __future__ import annotations

import pandas as pd

from iad.ml.clustering.dbscan import run_dbscan
from iad.ml.clustering.kmeans import run_kmeans
from iad.ml.clustering.reports import ClusteringReport


class ClusteringService:
    """Unsupervised clustering facade."""

    def kmeans(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str],
        n_clusters: int = 3,
        projection: str = "pca",
        random_state: int = 42,
    ) -> ClusteringReport:
        return run_kmeans(
            df,
            feature_columns=feature_columns,
            n_clusters=n_clusters,
            projection=projection,
            random_state=random_state,
        )

    def dbscan(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
        projection: str = "pca",
    ) -> ClusteringReport:
        return run_dbscan(
            df,
            feature_columns=feature_columns,
            eps=eps,
            min_samples=min_samples,
            projection=projection,
        )
