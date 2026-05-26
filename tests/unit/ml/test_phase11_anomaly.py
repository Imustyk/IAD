"""Phase 11 anomaly detection tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iad.ml.anomaly.isolation_forest import detect_isolation_forest
from iad.ml.anomaly.one_class_svm import detect_one_class_svm


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (100, 3))
    outlier = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    data = np.vstack([normal, outlier])
    return pd.DataFrame(data, columns=["a", "b", "c"])


@pytest.mark.unit
def test_isolation_forest(numeric_df) -> None:
    report = detect_isolation_forest(numeric_df, contamination=0.05)
    assert report.is_anomaly.sum() >= 1


@pytest.mark.unit
def test_one_class_svm(numeric_df) -> None:
    report = detect_one_class_svm(numeric_df, nu=0.05)
    assert len(report.scores) == len(numeric_df)
