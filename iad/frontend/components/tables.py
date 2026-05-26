"""Data table components with optional caching."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from iad.performance.fingerprints import dataframe_fingerprint


@st.cache_data(show_spinner=False, ttl=120)
def _cached_head(df: pd.DataFrame, n: int, fp: str) -> pd.DataFrame:
    return df.head(n)


def render_dataframe(
    df: pd.DataFrame,
    *,
    height: int | None = 400,
    use_container_width: bool = True,
    hide_index: bool = True,
    column_config: dict[str, Any] | None = None,
) -> None:
    """Render a styled dataframe with sensible defaults."""
    st.dataframe(
        df,
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
) -> None:
    """Show first *n* rows with caching for large frames."""
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
