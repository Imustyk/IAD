"""Drift metrics unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.preprocessing.drift.metrics import (
    chi_square_drift,
    jensen_shannon_divergence,
    ks_statistic,
    population_stability_index,
)


@pytest.mark.unit
def test_ks_statistic(iris_df) -> None:
    col = "sepal length (cm)"
    stat, pval = ks_statistic(iris_df[col], iris_df[col] * 1.05)
    assert 0 <= stat <= 1
    assert 0 <= pval <= 1


@pytest.mark.unit
def test_jensen_shannon(iris_df) -> None:
    dist = jensen_shannon_divergence(iris_df["species"], iris_df["species"])
    assert dist >= 0


@pytest.mark.unit
def test_psi_numeric() -> None:
    ref = pd.Series([1, 2, 3, 4, 5] * 20)
    cur = pd.Series([1, 2, 3, 4, 10] * 20)
    psi = population_stability_index(ref, cur, bins=5)
    assert psi >= 0


@pytest.mark.unit
def test_chi_square_drift(iris_df) -> None:
    stat, pval = chi_square_drift(iris_df["species"], iris_df["species"])
    assert stat >= 0
    assert 0 <= pval <= 1
