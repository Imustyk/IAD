"""Polars disabled / pandas fallback tests."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from tests.helpers.factories import csv_bytes


@pytest.mark.unit
def test_read_csv_pandas_fallback(monkeypatch) -> None:
    monkeypatch.setenv("IAD_PERF_USE_POLARS", "false")
    from iad.config.settings import get_settings

    get_settings.cache_clear()
    from iad.performance.polars_io import read_csv_fast

    df = read_csv_fast(io.BytesIO(csv_bytes()))
    assert isinstance(df, pd.DataFrame)
    get_settings.cache_clear()
