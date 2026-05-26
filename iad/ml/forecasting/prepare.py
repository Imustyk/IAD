"""Time-series preparation utilities."""
from __future__ import annotations

import pandas as pd

from iad.core.exceptions import SchemaError


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a column to timezone-naive datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            out = pd.to_datetime(series, errors="coerce")
        else:
            med = float(non_null.median())
            if med > 1e12:
                out = pd.to_datetime(series, errors="coerce", unit="ms")
            elif med > 1e9:
                out = pd.to_datetime(series, errors="coerce", unit="s")
            else:
                out = pd.to_datetime(series, errors="coerce")
    else:
        out = pd.to_datetime(series, errors="coerce", utc=True)

    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert(None)
    return out


def _coerce_value_series(series: pd.Series) -> pd.Series:
    """Parse a column to float for time-series values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": None, "nan": None, "None": None, "NA": None})
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _column_looks_like_datetime(series: pd.Series, *, min_valid_ratio: float) -> bool:
    """Heuristic: true date/time columns, not numeric IDs or arbitrary numbers."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return float(series.notna().mean()) >= min_valid_ratio

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        # Small integers (1, 2, 3 …) are IDs, not Unix epochs.
        if float(non_null.median()) < 1e9:
            return False
        parsed = _parse_datetime_series(series)
    else:
        parsed = _parse_datetime_series(series)

    valid = parsed.notna()
    if valid.sum() < 2 or valid.mean() < min_valid_ratio:
        return False
    unique_dates = parsed[valid].nunique()
    return unique_dates >= 2


def discover_datetime_columns(
    df: pd.DataFrame,
    *,
    min_valid_ratio: float = 0.5,
) -> list[str]:
    """Return column names that look like date/time (not IDs or plain metrics)."""
    return [
        col
        for col in df.columns
        if _column_looks_like_datetime(df[col], min_valid_ratio=min_valid_ratio)
    ]


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
    if datetime_column == value_column:
        raise SchemaError(
            "Datetime and value columns must differ.",
            user_message="Choose different datetime and value columns.",
        )

    frame = df[[datetime_column, value_column]].copy()
    n_raw = len(frame)

    frame[datetime_column] = _parse_datetime_series(frame[datetime_column])
    frame[value_column] = _coerce_value_series(frame[value_column])

    invalid_dates = int(frame[datetime_column].isna().sum())
    invalid_values = int(frame[value_column].isna().sum())
    frame = frame.dropna(subset=[datetime_column, value_column])

    if frame.empty:
        raise SchemaError(
            f"No valid rows after parsing dates/values (n={n_raw}, "
            f"bad_dates={invalid_dates}, bad_values={invalid_values}).",
            user_message=(
                f"Could not build a time series from '{datetime_column}' and '{value_column}'. "
                f"{invalid_dates} of {n_raw} rows failed date parsing — pick a real date/time column "
                f"(not categories like diagnosis or ID). "
                f"{invalid_values} rows had non-numeric values in the value column."
            ),
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
