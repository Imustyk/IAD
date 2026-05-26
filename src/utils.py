"""Shared helpers used across the application pages."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


SESSION_KEYS = {
    "dataset": "dataset",
    "dataset_name": "dataset_name",
    "business_case": "business_case",
    "model_bundle": "model_bundle",
    "preprocessor": "preprocessor",
    "target_column": "target_column",
    "task_type": "task_type",
    "feature_columns": "feature_columns",
    "training_report": "training_report",
}


def init_session_state() -> None:
    """Initialise default values in Streamlit session state."""
    defaults = {
        SESSION_KEYS["dataset"]: None,
        SESSION_KEYS["dataset_name"]: None,
        SESSION_KEYS["business_case"]: {
            "title": "",
            "problem": "",
            "objective": "",
            "kpis": "",
            "stakeholders": "",
            "data_sources": "",
        },
        SESSION_KEYS["model_bundle"]: None,
        SESSION_KEYS["preprocessor"]: None,
        SESSION_KEYS["target_column"]: None,
        SESSION_KEYS["task_type"]: None,
        SESSION_KEYS["feature_columns"]: None,
        SESSION_KEYS["training_report"]: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_dataset() -> pd.DataFrame | None:
    """Return the dataset from session state or render a friendly warning."""
    df = st.session_state.get(SESSION_KEYS["dataset"])
    if df is None:
        st.warning(
            "No dataset loaded yet. Open the **Data Loading** page first to "
            "upload a file, fetch a URL or pick a sample dataset."
        )
        return None
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def categorical_columns(df: pd.DataFrame, max_cardinality: int = 50) -> list[str]:
    cats: list[str] = []
    for col in df.columns:
        series = df[col]
        if series.dtype == "object" or str(series.dtype).startswith("category"):
            cats.append(col)
        elif pd.api.types.is_bool_dtype(series):
            cats.append(col)
        elif pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= max_cardinality:
            cats.append(col)
    return cats


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def coerce_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def humanize_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:,.1f} TB"


def detect_task_type(target: pd.Series) -> str:
    """Return 'classification' or 'regression' based on the target series."""
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or str(target.dtype).startswith("category")
    ):
        return "classification"
    nunique = target.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(target) and nunique <= 20:
        return "classification"
    if nunique <= 2:
        return "classification"
    return "regression"


def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """A describe() that always returns something even on empty frames."""
    if df.empty:
        return pd.DataFrame()
    try:
        return df.describe(include="all").transpose()
    except ValueError:
        return df.describe().transpose()


def percent_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / max(len(df), 1) * 100).round(2)
    out = pd.DataFrame({"missing": missing, "percent": pct})
    out = out.sort_values("missing", ascending=False)
    return out


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint of numeric columns where it is safe."""
    out = df.copy()
    for col in out.select_dtypes(include=["int"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def chunked(iterable, size: int):
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def cap_categories(series: pd.Series, top_n: int = 20) -> pd.Series:
    counts = series.value_counts(dropna=False)
    keep = set(counts.head(top_n).index)
    return series.where(series.isin(keep), other="Other")


def is_uniform(values: Iterable) -> bool:
    arr = np.asarray(list(values))
    return arr.size > 0 and np.all(arr == arr[0])
