"""Polars edge paths and fallbacks."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.factories import csv_bytes


@pytest.mark.unit
def test_read_parquet_bytes_buffer(iris_df, tmp_path: Path) -> None:
    from iad.performance.polars_io import read_parquet_fast

    path = tmp_path / "iris.parquet"
    iris_df.to_parquet(path)
    data = path.read_bytes()
    loaded = read_parquet_fast(io.BytesIO(data))
    assert len(loaded) == len(iris_df)


@pytest.mark.unit
def test_scan_lazy_csv(tmp_path: Path) -> None:
    from iad.performance.polars_io import scan_lazy_csv

    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    frame = scan_lazy_csv(path)
    if frame is not None:
        assert frame.collect().height == 2


@pytest.mark.unit
def test_read_parquet_pandas_fallback(tmp_path: Path, iris_df, monkeypatch) -> None:
    monkeypatch.setenv("IAD_PERF_USE_POLARS", "false")
    from iad.config.settings import get_settings

    get_settings.cache_clear()
    from iad.performance.polars_io import read_parquet_fast

    path = tmp_path / "iris.parquet"
    iris_df.to_parquet(path)
    loaded = read_parquet_fast(path)
    assert len(loaded) == len(iris_df)
    get_settings.cache_clear()


@pytest.mark.unit
def test_polars_unavailable_uses_pandas(monkeypatch) -> None:
    import iad.performance.polars_io as polars_io

    monkeypatch.setattr(polars_io, "_POLARS_AVAILABLE", False)
    monkeypatch.setenv("IAD_PERF_USE_POLARS", "true")
    from iad.config.settings import get_settings

    get_settings.cache_clear()
    df = polars_io.read_csv_fast(io.BytesIO(csv_bytes()))
    assert isinstance(df, pd.DataFrame)
    get_settings.cache_clear()
