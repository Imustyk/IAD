"""Dtype helper tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.preprocessing._dtypes import (
    categorical_columns,
    datetime_columns,
    is_categorical_like,
    numeric_columns,
)


@pytest.mark.unit
def test_categorical_detection(iris_df) -> None:
    assert is_categorical_like(iris_df["species"])
    assert "species" in categorical_columns(iris_df)


@pytest.mark.unit
def test_numeric_columns(iris_df) -> None:
    nums = numeric_columns(iris_df)
    assert len(nums) >= 4


@pytest.mark.unit
def test_datetime_columns() -> None:
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3)})
    assert "t" in datetime_columns(df)
