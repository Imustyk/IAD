"""Fingerprint tests."""
from __future__ import annotations

import pandas as pd

from iad.performance.fingerprints import dataframe_fingerprint, params_fingerprint


def test_dataframe_fingerprint_stable() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert dataframe_fingerprint(df) == dataframe_fingerprint(df)


def test_dataframe_fingerprint_changes_with_data() -> None:
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)


def test_params_fingerprint() -> None:
    assert params_fingerprint(a=1, b=2) == params_fingerprint(b=2, a=1)
