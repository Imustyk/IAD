"""Stable fingerprints for cache keys and data versioning."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    sample_rows: int = 500,
    extra: str | None = None,
) -> str:
    """Compute a stable hash for a DataFrame (shape + schema + sample).

    Uses a sample of rows so hashing stays fast on multi-million-row frames.
    """
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(str(c) for c in df.columns).encode())
    h.update(",".join(str(t) for t in df.dtypes.astype(str)).encode())
    if len(df) > 0:
        n = min(sample_rows, len(df))
        sample = df.head(n) if len(df) <= sample_rows * 2 else df.sample(n, random_state=0)
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()


def series_fingerprint(series: pd.Series) -> str:
    return dataframe_fingerprint(series.to_frame())


def params_fingerprint(**params: Any) -> str:
    """Hash arbitrary JSON-serialisable training / chart parameters."""
    import json

    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
