"""Shared test data factories."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def iris_dataframe() -> pd.DataFrame:
    raw = load_iris(as_frame=True)
    df = raw.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(raw.target_names)))
    return df


def regression_dataframe(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3.0 * x1 - 1.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def large_dataframe(n: int = 150_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.integers(0, 100, n),
        "b": rng.normal(0, 1, n),
        "cat": rng.choice(["x", "y", "z"], n),
    })


def csv_bytes() -> bytes:
    return b"col1,col2\n1,2\n3,4\n5,6\n"
