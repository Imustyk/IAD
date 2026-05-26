"""Memory optimisation — dtype downcasting, sampling, footprint reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.performance.memory")


@dataclass(frozen=True)
class MemoryFootprint:
    """Human-readable memory statistics for a DataFrame."""

    rows: int
    columns: int
    memory_bytes: int
    memory_mb: float
    dtypes: dict[str, str]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> MemoryFootprint:
        mem = int(df.memory_usage(deep=True).sum())
        return cls(
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            memory_bytes=mem,
            memory_mb=round(mem / (1024 * 1024), 2),
            dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        )


def optimize_dtypes(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """Downcast numeric columns and categorise low-cardinality strings.

    Typical savings: 30–70 % RAM on mixed-type business datasets.
    """
    out = df if inplace else df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_integer_dtype(series):
            out[col] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            out[col] = pd.to_numeric(series, downcast="float")
        elif series.dtype == object or str(series.dtype) == "string":
            nunique = series.nunique(dropna=True)
            if 0 < nunique < min(50, max(1, len(series) // 20)):
                out[col] = series.astype("category")
    return out


def sample_if_large(
    df: pd.DataFrame,
    *,
    max_rows: int | None = None,
    random_state: int = 42,
    strategy: Literal["head", "random"] = "random",
) -> tuple[pd.DataFrame, bool]:
    """Return ``(df, was_sampled)`` when row count exceeds threshold."""
    settings = get_settings()
    limit = max_rows or settings.PERF_LAZY_PREVIEW_ROWS
    if len(df) <= limit:
        return df, False
    if strategy == "head":
        sampled = df.head(limit).copy()
    else:
        sampled = df.sample(n=limit, random_state=random_state).copy()
    logger.info(
        "sampled dataframe %s -> %s rows (limit %s)",
        len(df),
        len(sampled),
        limit,
    )
    return sampled, True


def prepare_for_session(
    df: pd.DataFrame,
    *,
    optimize: bool | None = None,
) -> pd.DataFrame:
    """Standard pipeline when storing a dataset in Streamlit session state."""
    settings = get_settings()
    should_opt = optimize if optimize is not None else settings.PERF_AUTO_OPTIMIZE_DTYPES
    result = df
    if should_opt:
        result = optimize_dtypes(result)
    if len(result) > settings.PERF_POLARS_THRESHOLD_ROWS and settings.PERF_USE_POLARS:
        # Polars path already applied at load; only dtype optimisation here
        pass
    return result


def estimate_chunk_size(target_mb: float = 50) -> int:
    """Heuristic row chunk size for iterative processing."""
    return max(10_000, int(target_mb * 1024 * 1024 / 200))
