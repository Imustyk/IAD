"""DBSCAN density-based clustering."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.clustering.reduction import project_features
from iad.ml.clustering.reports import ClusteringReport

logger = get_logger("iad.ml.clustering.dbscan")


def run_dbscan(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    eps: float = 0.5,
    min_samples: int = 5,
    projection: str = "pca",
) -> ClusteringReport:
    """Cluster with DBSCAN; label -1 denotes noise."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Columns not found: {missing}",
            user_message="Select valid feature columns.",
        )

    matrix = df[feature_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matrix) < min_samples:
        raise AnalyticsError(
            "Not enough rows for DBSCAN min_samples.",
            user_message="Lower min_samples or add data.",
        )

    scaled = StandardScaler().fit_transform(matrix)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled)

    unique = set(labels)
    n_clusters = len(unique - {-1})
    metrics: dict[str, float] = {
        "noise_share": float((labels == -1).mean()),
        "n_clusters": float(n_clusters),
    }
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        metrics["silhouette"] = float(silhouette_score(scaled[mask], labels[mask]))

    label_series = pd.Series(labels, index=matrix.index, name="cluster")
    sizes = label_series.value_counts().sort_index().rename("count").reset_index()
    sizes.columns = ["cluster", "count"]

    coords = project_features(matrix, method=projection)
    coords["cluster"] = label_series

    logger.info("dbscan complete", extra={"eps": eps, "min_samples": min_samples})
    return ClusteringReport(
        method="dbscan",
        labels=label_series,
        n_clusters=n_clusters,
        metrics=metrics,
        projection=coords,
        projection_method=projection,
        feature_columns=feature_columns,
        cluster_sizes=sizes,
    )
