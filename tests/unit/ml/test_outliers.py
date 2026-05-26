"""Outlier detection quality module tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iad.ml.preprocessing.quality.outliers import (
    cap_outliers,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
)


@pytest.mark.unit
def test_iqr_detects_extreme() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3, 100]})
    report = detect_outliers_iqr(df, columns=["x"], k=1.5)
    assert report.method == "iqr"
    assert report.n_outliers[0] >= 1
    frame = report.to_frame()
    assert not frame.empty


@pytest.mark.unit
def test_zscore_detects_extreme() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 50), [20.0]])
    df = pd.DataFrame({"x": values})
    report = detect_outliers_zscore(df, columns=["x"], threshold=3.0)
    assert report.n_outliers[0] >= 1


@pytest.mark.unit
def test_isolation_forest(iris_df) -> None:
    numeric = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:3]
    report = detect_outliers_isolation_forest(iris_df, columns=numeric, contamination=0.1)
    assert report.method == "isolation_forest"


@pytest.mark.unit
def test_cap_outliers() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3, 100]})
    capped = cap_outliers(df, columns=["x"])
    assert capped["x"].max() < 100
