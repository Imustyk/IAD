"""Extended dtype helper coverage."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.preprocessing._dtypes import is_categorical_like


@pytest.mark.unit
def test_string_dtype_is_categorical() -> None:
    series = pd.Series(["a", "b", "c"], dtype="string")
    assert is_categorical_like(series)


@pytest.mark.unit
def test_datetime_not_categorical() -> None:
    series = pd.Series(pd.date_range("2024-01-01", periods=3))
    assert not is_categorical_like(series)


@pytest.mark.unit
def test_numeric_not_categorical() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    assert not is_categorical_like(series)
