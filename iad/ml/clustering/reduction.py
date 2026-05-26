"""Dimensionality reduction for cluster visualization."""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from iad.core.exceptions import AnalyticsError
from iad.core.logging import get_logger
from iad.ml.clustering.availability import umap_available

logger = get_logger("iad.ml.clustering.reduction")


def project_pca(
    matrix: pd.DataFrame,
    *,
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """PCA projection after standard scaling."""
    if matrix.shape[0] < 2:
        raise AnalyticsError(
            "Need at least 2 rows for PCA.",
            user_message="Add more rows to the dataset.",
        )
    n_components = min(n_components, matrix.shape[1], matrix.shape[0])
    scaled = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=n_components, random_state=random_state).fit_transform(scaled)
    cols = [f"pc{i + 1}" for i in range(coords.shape[1])]
    return pd.DataFrame(coords, index=matrix.index, columns=cols)


def project_umap(  # pragma: no cover — optional dependency; install `.[clustering]`
    matrix: pd.DataFrame,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """UMAP projection (optional dependency)."""
    if not umap_available():
        raise AnalyticsError(
            "umap-learn is not installed.",
            user_message="Install clustering extras: pip install umap-learn",
        )
    import umap

    if matrix.shape[0] < n_neighbors:
        n_neighbors = max(2, matrix.shape[0] - 1)

    scaled = StandardScaler().fit_transform(matrix)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    coords = reducer.fit_transform(scaled)
    cols = [f"umap{i + 1}" for i in range(coords.shape[1])]
    logger.info("umap projection", extra={"n_rows": len(matrix)})
    return pd.DataFrame(coords, index=matrix.index, columns=cols)


def project_features(
    matrix: pd.DataFrame,
    *,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    if method == "pca":
        return project_pca(matrix, n_components=n_components, random_state=random_state)
    if method == "umap":
        return project_umap(matrix, n_components=n_components, random_state=random_state)
    raise AnalyticsError(
        f"Unknown projection method {method!r}.",
        user_message="Choose pca or umap.",
    )
