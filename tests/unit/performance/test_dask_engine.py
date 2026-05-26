"""Dask engine unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.performance import dask_engine


@pytest.mark.unit
def test_dask_available() -> None:
    assert isinstance(dask_engine.dask_available(), bool)


@pytest.mark.unit
def test_dask_unavailable_flag(monkeypatch) -> None:
    monkeypatch.setattr(dask_engine, "_DASK_AVAILABLE", False)
    assert dask_engine.dask_available() is False


@pytest.mark.unit
def test_should_use_dask_small_frame(settings, iris_df) -> None:
    assert dask_engine.should_use_dask(iris_df) is False


@pytest.mark.unit
def test_should_use_dask_large_frame(settings) -> None:
    from tests.helpers.factories import large_dataframe

    df = large_dataframe(n=settings.PERF_DASK_THRESHOLD_ROWS + 100)
    assert dask_engine.should_use_dask(df) is True


@pytest.mark.unit
def test_describe_parallel_small(iris_df) -> None:
    out = dask_engine.describe_parallel(iris_df)
    assert isinstance(out, pd.DataFrame)
    assert not out.empty


@pytest.mark.unit
def test_value_counts_parallel(iris_df) -> None:
    out = dask_engine.value_counts_parallel(iris_df["species"], top_n=3)
    assert list(out.columns) == ["species", "count", "percent"]
    assert len(out) <= 3


@pytest.mark.unit
def test_aggregate_numeric_empty() -> None:
    result = dask_engine.aggregate_numeric(pd.DataFrame(), [])
    assert result.empty


@pytest.mark.unit
def test_aggregate_numeric_small(iris_df) -> None:
    cols = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"]
    result = dask_engine.aggregate_numeric(iris_df, cols[:2], agg="mean")
    assert len(result) == 2


@pytest.mark.unit
def test_to_dask_dataframe(settings) -> None:
    from tests.helpers.factories import large_dataframe

    df = large_dataframe(n=settings.PERF_DASK_THRESHOLD_ROWS + 50)
    ddf = dask_engine.to_dask_dataframe(df, npartitions=4)
    assert ddf.npartitions >= 2
