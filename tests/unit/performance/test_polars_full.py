"""Extended Polars I/O tests."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from iad.performance.polars_io import read_csv_fast, read_parquet_fast
from tests.helpers.factories import csv_bytes


@pytest.mark.unit
def test_read_csv_bytes() -> None:
    df = read_csv_fast(io.BytesIO(csv_bytes()))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


@pytest.mark.unit
def test_read_csv_path(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_bytes(csv_bytes())
    df = read_csv_fast(path)
    assert len(df) == 3


@pytest.mark.unit
def test_read_parquet_roundtrip(tmp_path: Path, iris_df) -> None:
    path = tmp_path / "iris.parquet"
    iris_df.to_parquet(path)
    loaded = read_parquet_fast(path)
    pd.testing.assert_frame_equal(loaded, iris_df, check_dtype=False)
