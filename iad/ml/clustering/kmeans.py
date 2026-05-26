"""K-Means clustering."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.clustering.reduction import project_features
from iad.ml.clustering.reports import ClusteringReport

logger = get_logger("iad.ml.clustering.kmeans")


def run_kmeans(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    n_clusters: int = 3,
    projection: str = "pca",
    random_state: int = 42,
) -> ClusteringReport:
    """Cluster numeric features with K-Means."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Columns not found: {missing}",
            user_message="Select valid feature columns.",
        )

    matrix = df[feature_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matrix) < n_clusters:
        raise AnalyticsError(
            f"Need at least {n_clusters} rows for K-Means.",
            user_message="Reduce k or add more data.",
        )

    scaled = StandardScaler().fit_transform(matrix)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(scaled)

    metrics: dict[str, float] = {
        "inertia": float(model.inertia_),
    }
    if len(set(labels)) > 1:
        metrics["silhouette"] = float(silhouette_score(scaled, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(scaled, labels))

    label_series = pd.Series(labels, index=matrix.index, name="cluster")
    sizes = label_series.value_counts().sort_index().rename("count").reset_index()
    sizes.columns = ["cluster", "count"]

    coords = project_features(matrix, method=projection, random_state=random_state)
    coords["cluster"] = label_series

    logger.info("kmeans complete", extra={"k": n_clusters, "n_rows": len(matrix)})
    return ClusteringReport(
        method="kmeans",
        labels=label_series,
        n_clusters=n_clusters,
        metrics=metrics,
        projection=coords,
        projection_method=projection,
        feature_columns=feature_columns,
        cluster_sizes=sizes,
    )
