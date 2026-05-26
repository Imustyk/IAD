"""High-performance data loading bridge (Polars + memory optimisation)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from iad.config.settings import get_settings
from iad.performance.memory import MemoryFootprint, prepare_for_session
from iad.performance.polars_io import read_csv_fast, read_parquet_fast


def load_uploaded_file_fast(uploaded: object) -> pd.DataFrame:
    """Load a Streamlit ``UploadedFile`` with the fastest available backend."""
    name = getattr(uploaded, "name", "upload.csv")
    ext = Path(name).suffix.lower()
    raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
    buffer = io.BytesIO(raw)

    if ext in (".parquet",):
        df = read_parquet_fast(buffer)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(buffer)
    elif ext in (".json",):
        df = pd.read_json(buffer)
    else:
        sep = "\t" if ext in (".tsv", ".txt") else ","
        df = read_csv_fast(buffer, sep=sep)

    return prepare_for_session(df)


def load_from_path_fast(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = read_parquet_fast(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        df = read_csv_fast(path)
    return prepare_for_session(df)


def loading_metadata(df: pd.DataFrame) -> dict[str, object]:
    fp = MemoryFootprint.from_dataframe(df)
    settings = get_settings()
    return {
        "rows": fp.rows,
        "columns": fp.columns,
        "memory_mb": fp.memory_mb,
        "polars_enabled": settings.PERF_USE_POLARS,
        "dask_eligible": fp.rows >= settings.PERF_DASK_THRESHOLD_ROWS,
    }
