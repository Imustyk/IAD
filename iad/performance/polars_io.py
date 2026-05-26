"""Polars-accelerated I/O with pandas fallback."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.performance.polars")

_POLARS_AVAILABLE: bool | None = None


def polars_available() -> bool:
    global _POLARS_AVAILABLE
    if _POLARS_AVAILABLE is None:
        try:
            import polars as pl  # noqa: F401

            _POLARS_AVAILABLE = True
        except ImportError:
            _POLARS_AVAILABLE = False
    return _POLARS_AVAILABLE


def _pandas_csv_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Normalize CSV keyword arguments for Polars vs pandas."""
    out = dict(kwargs)
    if "sep" in out and "separator" not in out:
        out["separator"] = out.pop("sep")
    return out


def read_csv_fast(path_or_buffer: str | Path | BinaryIO, **kwargs: object) -> pd.DataFrame:
    """Read CSV via Polars when enabled and available, else pandas."""
    settings = get_settings()
    if settings.PERF_USE_POLARS and polars_available():
        import polars as pl

        pl_kwargs = _pandas_csv_kwargs(dict(kwargs))
        logger.debug("reading CSV with Polars")
        if isinstance(path_or_buffer, (str, Path)):
            lf = pl.read_csv(path_or_buffer, **pl_kwargs)  # type: ignore[arg-type]
        else:
            if hasattr(path_or_buffer, "read"):
                data = path_or_buffer.read()
                if hasattr(path_or_buffer, "seek"):
                    path_or_buffer.seek(0)
            else:
                data = path_or_buffer
            lf = pl.read_csv(io.BytesIO(data), **pl_kwargs)  # type: ignore[arg-type]
        return lf.to_pandas()
    if isinstance(path_or_buffer, (str, Path)):
        return pd.read_csv(path_or_buffer, **kwargs)  # type: ignore[arg-type]
    return pd.read_csv(path_or_buffer, **kwargs)  # type: ignore[arg-type]


def read_parquet_fast(path_or_buffer: str | Path | BinaryIO) -> pd.DataFrame:
    settings = get_settings()
    if settings.PERF_USE_POLARS and polars_available():
        import polars as pl

        logger.debug("reading Parquet with Polars")
        if isinstance(path_or_buffer, (str, Path)):
            return pl.read_parquet(path_or_buffer).to_pandas()
        data = path_or_buffer.read() if hasattr(path_or_buffer, "read") else path_or_buffer
        return pl.read_parquet(io.BytesIO(data)).to_pandas()
    if isinstance(path_or_buffer, (str, Path)):
        return pd.read_parquet(path_or_buffer)
    return pd.read_parquet(path_or_buffer)


def scan_lazy_csv(path: str | Path) -> object | None:
    """Return a Polars ``LazyFrame`` for out-of-core queries, or None."""
    if not polars_available():
        return None
    settings = get_settings()
    if not settings.PERF_USE_POLARS:
        return None
    import polars as pl

    return pl.scan_csv(path)
