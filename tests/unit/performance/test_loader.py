"""Performance loader bridge tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from iad.performance.loader import load_from_path_fast, load_uploaded_file_fast, loading_metadata
from tests.helpers.factories import csv_bytes


@pytest.mark.unit
def test_load_uploaded_csv() -> None:
    uploaded = MagicMock()
    uploaded.name = "data.csv"
    uploaded.read.return_value = csv_bytes()
    df = load_uploaded_file_fast(uploaded)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


@pytest.mark.unit
def test_load_uploaded_tsv() -> None:
    uploaded = MagicMock()
    uploaded.name = "data.tsv"
    uploaded.read.return_value = b"a\tb\n1\t2\n"
    df = load_uploaded_file_fast(uploaded)
    assert df.shape[1] == 2


@pytest.mark.unit
def test_load_from_path_csv(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_bytes(csv_bytes())
    df = load_from_path_fast(path)
    assert len(df) == 3


@pytest.mark.unit
def test_load_from_path_json(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text('[{"x": 1}, {"x": 2}]')
    df = load_from_path_fast(path)
    assert "x" in df.columns


@pytest.mark.unit
def test_loading_metadata(iris_df) -> None:
    meta = loading_metadata(iris_df)
    assert meta["rows"] == len(iris_df)
    assert meta["columns"] == len(iris_df.columns)
    assert "memory_mb" in meta
