"""Shared dtype helpers.

Modern pandas (>=2.1) introduced first-class ``str`` and ``StringDtype``
columns that ``df[col].dtype == "object"`` no longer matches. The previous
code used the legacy check in several places; this module centralises the
correct, future-proof detection so we never drift again.
"""
from __future__ import annotations

import pandas as pd


def is_categorical_like(series: pd.Series) -> bool:
    """True for object / string / categorical / boolean columns."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    if pd.api.types.is_categorical_dtype(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return False


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_categorical_like(df[c])]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


__all__ = ["is_categorical_like", "categorical_columns", "numeric_columns", "datetime_columns"]
