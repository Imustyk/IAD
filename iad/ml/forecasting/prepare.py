"""Time-series preparation utilities."""
from __future__ import annotations

import pandas as pd

from iad.core.exceptions import SchemaError


def prepare_series(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    value_column: str,
    freq: str | None = None,
) -> pd.Series:
    """Return a sorted, datetime-indexed univariate series."""
    if datetime_column not in df.columns or value_column not in df.columns:
        raise SchemaError(
            "Datetime or value column missing.",
            user_message="Select valid datetime and value columns.",
        )

    frame = df[[datetime_column, value_column]].copy()
    frame[datetime_column] = pd.to_datetime(frame[datetime_column], errors="coerce")
    frame = frame.dropna(subset=[datetime_column, value_column])
    if frame.empty:
        raise SchemaError(
            "No valid rows after parsing dates.",
            user_message="Check datetime parsing and missing values.",
        )

    series = (
        frame.sort_values(datetime_column)
        .drop_duplicates(subset=[datetime_column], keep="last")
        .set_index(datetime_column)[value_column]
        .astype(float)
    )
    if freq:
        series = series.asfreq(freq)
    return series
