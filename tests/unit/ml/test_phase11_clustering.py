"""Phase 11 clustering tests."""
from __future__ import annotations

import pytest

from iad.ml.clustering.dbscan import run_dbscan
from iad.ml.clustering.kmeans import run_kmeans


@pytest.mark.unit
def test_kmeans(iris_df) -> None:
    features = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:4]
    report = run_kmeans(iris_df, feature_columns=features, n_clusters=3)
    assert report.n_clusters == 3
    assert report.projection is not None


@pytest.mark.unit
def test_dbscan(iris_df) -> None:
    features = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:4]
    report = run_dbscan(iris_df, feature_columns=features, eps=1.5, min_samples=3)
    assert report.method == "dbscan"
