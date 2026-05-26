"""Descriptive analytics page."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from iad.frontend.components.charts import render_plotly
from iad.frontend.components.metric_cards import MetricSpec, render_metric_row
from iad.frontend.components.tables import render_dataframe
from iad.config import get_settings
from iad.frontend.layouts.page import setup_page
from iad.performance.dask_engine import should_use_dask, value_counts_parallel
from iad.performance.memory import MemoryFootprint
from src.descriptive import (
    categorical_summary,
    distribution_normality,
    missing_value_report,
    summary_statistics,
    time_series_summary,
    value_counts_table,
)
from src.utils import (
    SESSION_KEYS,
    categorical_columns,
    datetime_columns,
    numeric_columns,
    require_dataset,
)


setup_page(
    "Descriptive Analysis",
    caption="Step 2 — univariate statistics, distributions, value counts and missing-data analysis.",
)

df = require_dataset()
if df is None:
    st.stop()

_settings = get_settings()
_fp = MemoryFootprint.from_dataframe(df)
if _fp.memory_mb > 50 or should_use_dask(df):
    st.caption(
        f"Dataset footprint: **{_fp.memory_mb} MB** · "
        f"{'Dask parallel path available' if should_use_dask(df) else 'in-memory analytics'}"
    )

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)
date_cols = datetime_columns(df)

st.markdown(
    f"**{df.shape[0]:,}** rows · **{df.shape[1]:,}** columns · "
    f"{len(num_cols)} numeric · {len(cat_cols)} categorical · "
    f"{len(date_cols)} datetime"
)

missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
render_metric_row([
    MetricSpec("Total cells", f"{df.size:,}"),
    MetricSpec("Missing cells", f"{int(df.isna().sum().sum()):,}", delta=f"{missing_pct:.1f}%"),
    MetricSpec("Duplicate rows", f"{int(df.duplicated().sum()):,}"),
    MetricSpec("Numeric features", f"{len(num_cols)}"),
])


tab_stats, tab_dist, tab_cats, tab_missing, tab_ts = st.tabs(
    [
        "Summary statistics",
        "Distributions",
        "Categorical features",
        "Missing values",
        "Time series",
    ]
)


with tab_stats:
    st.subheader("Numeric summary")
    stats = summary_statistics(df)
    if stats.empty:
        st.info("No numeric columns to summarise.")
    else:
        render_dataframe(stats)

    if cat_cols:
        st.subheader("Categorical summary")
        st.dataframe(categorical_summary(df, cat_cols), use_container_width=True)


with tab_dist:
    if not num_cols:
        st.info("No numeric columns to plot.")
    else:
        col = st.selectbox("Numeric column", num_cols)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col, marginal="box", nbins=40, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.violin(df, y=col, box=True, points="outliers", title=f"Spread of {col}")
            st.plotly_chart(fig, use_container_width=True)

        norm = distribution_normality(df, col)
        st.markdown("**Normality test**")
        st.json(norm)

        if cat_cols:
            group_col = st.selectbox("Compare across category", ["—"] + cat_cols, index=0)
            if group_col != "—":
                fig = px.box(df, x=group_col, y=col, points="outliers",
                             title=f"{col} by {group_col}")
                st.plotly_chart(fig, use_container_width=True)


with tab_cats:
    if not cat_cols:
        st.info("No categorical columns detected.")
    else:
        col = st.selectbox("Categorical column", cat_cols)
        top_n = st.slider("Top N categories", 5, 50, 20, 5)
        if should_use_dask(df):
            vc = value_counts_parallel(df[col], top_n=top_n, total_rows=len(df))
        else:
            vc = value_counts_table(df, col, top_n=top_n)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(vc, use_container_width=True)
        with c2:
            fig = px.bar(vc, x=col, y="count", title=f"{col} — top {top_n} values",
                         text="percent")
            st.plotly_chart(fig, use_container_width=True)

        if num_cols:
            num_for_group = st.selectbox("Average a numeric column per category", num_cols)
            agg = (
                df.groupby(col, dropna=False)[num_for_group]
                .agg(["mean", "median", "count"])
                .reset_index()
                .head(top_n)
            )
            fig = px.bar(agg, x=col, y="mean",
                         title=f"Average {num_for_group} by {col}")
            st.plotly_chart(fig, use_container_width=True)


with tab_missing:
    report = missing_value_report(df)
    st.dataframe(report, use_container_width=True)
    if report["missing"].sum() > 0:
        plot_data = report.head(30)
        fig = px.bar(plot_data, x="column", y="percent",
                     title="Missing values per column (%)",
                     color="dtype")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected.")


with tab_ts:
    if not date_cols or not num_cols:
        st.info(
            "Time-series view needs at least one datetime column and one numeric column. "
            "Use the **Data hygiene** section on the Data Loading page to parse dates."
        )
    else:
        date_col = st.selectbox("Datetime column", date_cols)
        value_col = st.selectbox("Numeric column", num_cols)
        ts = time_series_summary(df, date_col, value_col)
        if ts.empty:
            st.info("Not enough valid date/value pairs for a time series.")
        else:
            fig = px.line(ts, x=date_col, y="mean",
                          title=f"Monthly mean of {value_col}")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.bar(ts, x=date_col, y="sum", title=f"Monthly sum of {value_col}")
            st.plotly_chart(fig2, use_container_width=True)
