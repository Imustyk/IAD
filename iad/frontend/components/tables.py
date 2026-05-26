"""Data table components with optional caching."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from iad.performance.fingerprints import dataframe_fingerprint


@st.cache_data(show_spinner=False, ttl=120)
def _cached_head(df: pd.DataFrame, n: int, fp: str) -> pd.DataFrame:
    return df.head(n)


def format_schema_sample(series: pd.Series) -> str:
    """First non-null value as a display string for schema tables."""
    non_null = series.dropna()
    if non_null.empty:
        return ""
    return _format_cell_sample(non_null.iloc[0])


def _format_cell_sample(value: object) -> str:
    """One-line string for schema preview cells (Arrow-safe)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_cell_sample(v) for v in value)[:200]
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns so ``st.dataframe`` / PyArrow can serialize mixed types."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_extension_array_dtype(series.dtype):
            out[col] = series.astype(str)
            continue
        if series.dtype == object:
            out[col] = series.map(_format_cell_sample)
        elif pd.api.types.is_datetime64_any_dtype(series):
            out[col] = series.astype(str)
    return out


def render_dataframe(
    df: pd.DataFrame,
    *,
    height: int | None = 400,
    use_container_width: bool = True,
    hide_index: bool = True,
    column_config: dict[str, Any] | None = None,
    arrow_safe: bool = True,
) -> None:
    """Render a styled dataframe with sensible defaults."""
    display = arrow_safe_dataframe(df) if arrow_safe else df
    st.dataframe(
        display,
        height=height,
        use_container_width=use_container_width,
        hide_index=hide_index,
        column_config=column_config or {},
    )


def render_preview(
    df: pd.DataFrame,
    n: int = 10,
    *,
    title: str | None = "Data preview",
    empty_message: str | None = None,
) -> None:
    """Show first *n* rows with caching for large frames."""
    if df is None or df.empty:
        from iad.frontend.components.ui import render_empty_state

        render_empty_state(
            title or "No data",
            empty_message or "There is nothing to display yet.",
            hint="Load a dataset from Data loading.",
        )
        return
    if title:
        st.markdown(f"**{title}**")
    preview = _cached_head(df, n, dataframe_fingerprint(df))
    render_dataframe(preview, height=min(35 * (n + 1), 400))


def render_summary_table(
    summary: pd.DataFrame,
    *,
    title: str | None = None,
) -> None:
    if title:
        st.markdown(f"**{title}**")
    render_dataframe(summary.reset_index() if summary.index.name else summary)


def render_download_csv(
    df: pd.DataFrame,
    filename: str = "export.csv",
    *,
    label: str = "Download CSV",
) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")
