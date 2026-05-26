"""Load pandas DataFrames from API uploads (bytes + filename)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from iad.core.exceptions import DataLoadError
from iad.core.logging import get_logger
from iad.performance.loader import loading_metadata
from iad.performance.memory import prepare_for_session
from iad.performance.polars_io import read_csv_fast, read_parquet_fast

logger = get_logger("iad.backend.dataframe_io")


def load_dataframe_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame."""
    if not data:
        raise DataLoadError("Empty upload.", user_message="The uploaded file is empty.")
    buffer = io.BytesIO(data)
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".parquet":
            df = read_parquet_fast(buffer)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(buffer)
        elif ext == ".json":
            df = pd.read_json(buffer)
        else:
            sep = "\t" if ext in (".tsv", ".txt") else ","
            df = read_csv_fast(buffer, sep=sep)
    except Exception as exc:
        logger.exception("failed to parse upload %s", filename)
        raise DataLoadError(
            f"Could not parse {filename}: {exc}",
            user_message="Failed to read the uploaded file. Check format and encoding.",
        ) from exc

    if df.empty:
        raise DataLoadError(
            "Uploaded file has no rows.",
            user_message="The uploaded file contains no data rows.",
        )
    prepared = prepare_for_session(df)
    meta = loading_metadata(prepared)
    logger.info(
        "dataframe loaded from upload",
        extra={"ctx_filename": filename, "ctx_rows": meta["rows"], "ctx_columns": meta["columns"]},
    )
    return prepared
