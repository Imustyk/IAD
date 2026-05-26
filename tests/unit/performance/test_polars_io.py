"""Polars I/O tests — skip when polars not installed."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from iad.performance.polars_io import polars_available, read_csv_fast

pytest.importorskip("polars")


def test_read_csv_fast_roundtrip() -> None:
    assert polars_available()
    csv = "a,b\n1,2\n3,4\n"
    df = read_csv_fast(io.BytesIO(csv.encode()))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
