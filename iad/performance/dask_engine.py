"""Dask-backed operations for datasets above the row threshold."""
from __future__ import annotations

from typing import Any

import pandas as pd

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.performance.dask")

_DASK_AVAILABLE: bool | None = None


def dask_available() -> bool:
    global _DASK_AVAILABLE
    if _DASK_AVAILABLE is None:
        try:
            import dask.dataframe as dd  # noqa: F401

            _DASK_AVAILABLE = True
        except ImportError:
            _DASK_AVAILABLE = False
    return _DASK_AVAILABLE


def should_use_dask(df: pd.DataFrame) -> bool:
    settings = get_settings()
    return (
        settings.PERF_USE_DASK
        and dask_available()
        and len(df) >= settings.PERF_DASK_THRESHOLD_ROWS
    )


def to_dask_dataframe(df: pd.DataFrame, npartitions: int | None = None) -> Any:
    """Partition an in-memory pandas frame for parallel apply/compute."""
    import dask.dataframe as dd

    settings = get_settings()
    parts = npartitions or max(2, len(df) // settings.PERF_DASK_PARTITION_ROWS)
    parts = min(parts, 32)
    logger.debug("creating dask dataframe with %s partitions", parts)
    return dd.from_pandas(df, npartitions=parts)


def describe_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Parallel ``describe()`` for large numeric frames."""
    if not should_use_dask(df):
        return df.describe(include="all").transpose()

    ddf = to_dask_dataframe(df)
    num = ddf.describe().compute()
    return num.transpose() if hasattr(num, "transpose") else num


def value_counts_parallel(
    series: pd.Series,
    top_n: int = 20,
    *,
    total_rows: int | None = None,
) -> pd.DataFrame:
    """Top-N value counts via Dask for long columns (matches ``value_counts_table`` shape)."""
    col = series.name or "value"
    n = total_rows if total_rows is not None else len(series)

    if not should_use_dask(series.to_frame()):
        counts = series.value_counts(dropna=False).head(top_n)
        out = counts.rename_axis(col).reset_index(name="count")
        out["percent"] = (out["count"] / max(n, 1) * 100).round(2)
        return out

    ddf = to_dask_dataframe(series.to_frame())
    counts = ddf[col].value_counts().compute().head(top_n)
    out = counts.rename_axis(col).reset_index(name="count")
    out["percent"] = (out["count"] / max(n, 1) * 100).round(2)
    return out


def aggregate_numeric(
    df: pd.DataFrame,
    columns: list[str],
    agg: str = "mean",
) -> pd.Series:
    """Run a simple aggregation (mean/sum/min/max) in parallel when large."""
    if not columns:
        return pd.Series(dtype=float)
    subset = df[columns]
    if not should_use_dask(subset):
        return getattr(subset, agg)()

    ddf = to_dask_dataframe(subset)
    computed = getattr(ddf, agg)().compute()
    return computed
