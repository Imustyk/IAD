"""Data Loading page — uploads, URLs and sample datasets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from iad.frontend.components.metric_cards import MetricSpec, render_metric_row
from iad.frontend.components.tables import format_schema_sample, render_dataframe, render_preview
from iad.frontend.components import alerts
from iad.frontend.components.uploaders import url_loader
from iad.config import get_settings
from iad.frontend.layouts.page import setup_page
from iad.frontend.services.context import store_dataset
from iad.performance.lazy import LazyDatasetView
from iad.performance.memory import MemoryFootprint
from iad.state.session import KEY_DATASET, KEY_DATASET_NAME, KEY_TARGET_COLUMN
from src.data_loader import (
    SAMPLE_DATASETS,
    load_from_url,
    load_sample,
    load_uploaded_file,
)
from src.utils import coerce_datetime, humanize_bytes

settings = get_settings()

setup_page(
    "Data Loading",
    caption="Step 1 — bring data in from a file, a public URL or a curated sample dataset.",
)

tab_sample, tab_upload, tab_url = st.tabs([
    "Sample datasets",
    "Upload file",
    "From URL",
])


def _on_load(df: pd.DataFrame, name: str) -> None:
    store_dataset(df, name)
    alerts.success(f"Loaded **{name}** ({df.shape[0]:,} rows × {df.shape[1]:,} columns).")


with tab_sample:
    st.markdown("Choose a curated sample dataset to explore the platform without uploading.")
    name = st.selectbox(
        "Sample dataset",
        list(SAMPLE_DATASETS.keys()),
        index=5,
    )
    sample = SAMPLE_DATASETS[name]
    st.info(f"**{sample.name}** — {sample.description}\n\nSuggested target: `{sample.suggested_target}`")
    if st.button("Load this dataset", type="primary"):
        df = load_sample(name)
        store_dataset(df, name, suggested_target=sample.suggested_target)
        st.session_state[KEY_TARGET_COLUMN] = sample.suggested_target
        alerts.success(f"Loaded **{name}** ({df.shape[0]:,} rows × {df.shape[1]:,} cols).")


with tab_upload:
    uploaded = st.file_uploader(
        "Upload a CSV, TSV, Excel, JSON or Parquet file",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        try:
            if settings.PERF_USE_POLARS:
                from iad.frontend.services.dataset import load_uploaded

                view = load_uploaded(uploaded)
                alerts.success(
                    f"Loaded **{uploaded.name}** — {view.summary_line()} "
                    f"(Polars-accelerated path)."
                )
            else:
                df = load_uploaded_file(uploaded)
                _on_load(df, uploaded.name)
            st.caption(f"File size: {humanize_bytes(uploaded.size)}")
        except Exception as exc:
            alerts.error(f"Failed to read the file: {exc}", show_details=True, exc=exc)


with tab_url:
    df = url_loader(
        loader=load_from_url,
        on_success=lambda d, u: _on_load(d, u.split("/")[-1] or u),
    )


st.divider()

df = st.session_state.get(KEY_DATASET)
if df is None:
    alerts.info("Load a dataset above to see a preview here.")
    st.stop()

st.subheader(f"Preview — {st.session_state.get(KEY_DATASET_NAME, 'Dataset')}")

footprint = MemoryFootprint.from_dataframe(df)
view = LazyDatasetView.from_dataframe(df)
if view.was_sampled:
    st.caption(f"Memory: {footprint.memory_mb} MB · large dataset preview ({view.summary_line()})")

missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
render_metric_row([
    MetricSpec("Rows", f"{df.shape[0]:,}"),
    MetricSpec("Columns", f"{df.shape[1]:,}"),
    MetricSpec("Missing cells", f"{int(df.isna().sum().sum()):,}", delta=f"{missing_pct:.1f}%"),
    MetricSpec("Duplicate rows", f"{int(df.duplicated().sum()):,}"),
])

render_preview(df, n=50, title=None)

with st.expander("Schema & data types"):
    schema = pd.DataFrame({
        "column": df.columns.astype(str),
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().astype(int).values,
        "unique": df.nunique(dropna=True).astype(int).values,
        "sample": [format_schema_sample(df[c]) for c in df.columns],
    })
    render_dataframe(schema, height=min(35 * (len(schema) + 1), 480))

with st.expander("Data hygiene"):
    cols_to_parse = st.multiselect(
        "Parse these columns as datetime",
        [c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c])],
    )
    drop_dupes = st.checkbox("Drop duplicate rows", value=False)
    drop_missing = st.checkbox("Drop columns with > X% missing", value=False)
    threshold = 50
    if drop_missing:
        threshold = st.slider("Missing % threshold", 10, 90, 50, 5)
    if st.button("Apply hygiene operations"):
        cleaned = df.copy()
        if cols_to_parse:
            cleaned = coerce_datetime(cleaned, cols_to_parse)
        if drop_dupes:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            st.write(f"Dropped {before - len(cleaned):,} duplicate rows.")
        if drop_missing:
            keep = [c for c in cleaned.columns if cleaned[c].isna().mean() * 100 <= threshold]
            removed = [c for c in cleaned.columns if c not in keep]
            cleaned = cleaned[keep]
            if removed:
                st.write(f"Dropped {len(removed)} sparse columns: {removed}")
        st.session_state[KEY_DATASET] = cleaned
        alerts.success("Hygiene operations applied.")
        st.rerun()
