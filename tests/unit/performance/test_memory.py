"""Memory optimisation tests."""
from __future__ import annotations

import pandas as pd

from iad.performance.memory import MemoryFootprint, optimize_dtypes, sample_if_large


def test_optimize_dtypes_downcasts_integers() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.5, 3.1]})
    optimised = optimize_dtypes(df)
    assert str(optimised["a"].dtype) in ("int8", "int16", "int32", "Int8", "Int16", "Int32")


def test_sample_if_large() -> None:
    df = pd.DataFrame({"x": range(10_000)})
    sampled, was = sample_if_large(df, max_rows=1000)
    assert was is True
    assert len(sampled) == 1000


def test_memory_footprint() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    fp = MemoryFootprint.from_dataframe(df)
    assert fp.rows == 3
    assert fp.memory_mb >= 0
