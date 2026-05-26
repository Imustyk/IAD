"""Diagnostic analytics page — correlations, group comparisons, hypothesis tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from iad.frontend.components.charts import render_plotly
from iad.frontend.layouts.page import setup_page
from src.diagnostic import (
    anova_test,
    chi_square_test,
    correlation_matrix,
    group_comparison,
    outliers_overview,
    t_test,
    top_correlations,
)
from src.utils import (
    SESSION_KEYS,
    categorical_columns,
    numeric_columns,
    require_dataset,
)


setup_page(
    "Diagnostic Analysis",
    icon="🔍",
    caption="Step 3 — correlations, group differences, hypothesis tests and outliers.",
)

df = require_dataset()
if df is None:
    st.stop()

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)


tab_corr, tab_groups, tab_tests, tab_outliers, tab_pairs = st.tabs(
    [
        "🧲 Correlations",
        "👥 Group comparisons",
        "🧪 Hypothesis tests",
        "🚩 Outliers",
        "🔗 Pairwise plots",
    ]
)


with tab_corr:
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for correlation analysis.")
    else:
        method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = correlation_matrix(df, method=method)  # type: ignore[arg-type]
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"{method.title()} correlation matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        target_options = ["—"] + num_cols
        target = st.selectbox(
            "Focus on a target column (optional)",
            target_options,
            index=target_options.index(st.session_state[SESSION_KEYS["target_column"]])
            if st.session_state[SESSION_KEYS["target_column"]] in target_options else 0,
        )
        top_n = st.slider("Top N", 5, 30, 10, 1)
        if target == "—":
            top = top_correlations(df, target=None, top_n=top_n, method=method)
        else:
            top = top_correlations(df, target=target, top_n=top_n, method=method)
        st.dataframe(top, use_container_width=True)


with tab_groups:
    if not cat_cols or not num_cols:
        st.info("Need at least one categorical column and one numeric column.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Group by", cat_cols)
        with c2:
            value_col = st.selectbox("Numeric value", num_cols)
        comparison = group_comparison(df, group_col, value_col)
        st.dataframe(comparison, use_container_width=True)
        if not comparison.empty:
            fig = px.bar(
                comparison,
                x=group_col, y="mean",
                error_y="std",
                title=f"Mean {value_col} per {group_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(df, x=group_col, y=value_col, points="outliers",
                          title=f"Distribution of {value_col} per {group_col}")
            st.plotly_chart(fig2, use_container_width=True)


with tab_tests:
    st.markdown(
        "Run formal statistical tests to confirm or reject hypotheses about "
        "differences and dependencies in the data."
    )
    test_kind = st.radio(
        "Test type",
        ["t-test (numeric vs binary)", "ANOVA (numeric vs multi-class)", "Chi-square (categorical vs categorical)"],
        horizontal=True,
    )

    if test_kind.startswith("t-test"):
        if not cat_cols or not num_cols:
            st.info("Need a binary categorical column and a numeric column.")
        else:
            binary_cats = [c for c in cat_cols if df[c].nunique(dropna=True) == 2]
            if not binary_cats:
                st.info("No binary categorical column found.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    group_col = st.selectbox("Binary group column", binary_cats, key="ttest_g")
                with c2:
                    value_col = st.selectbox("Numeric column", num_cols, key="ttest_v")
                if st.button("Run t-test"):
                    st.json(t_test(df, group_col, value_col))

    elif test_kind.startswith("ANOVA"):
        if not cat_cols or not num_cols:
            st.info("Need a categorical column and a numeric column.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                group_col = st.selectbox("Group column", cat_cols, key="anova_g")
            with c2:
                value_col = st.selectbox("Numeric column", num_cols, key="anova_v")
            if st.button("Run ANOVA"):
                st.json(anova_test(df, group_col, value_col))

    else:
        if len(cat_cols) < 2:
            st.info("Need at least two categorical columns.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                col_a = st.selectbox("Column A", cat_cols, key="chi_a")
            with c2:
                col_b = st.selectbox("Column B", [c for c in cat_cols if c != col_a], key="chi_b")
            if st.button("Run Chi-square"):
                result = chi_square_test(df, col_a, col_b)
                st.json(result)
                contingency = pd.crosstab(df[col_a], df[col_b])
                fig = px.imshow(contingency, text_auto=True, aspect="auto",
                                title=f"Contingency table: {col_a} × {col_b}")
                st.plotly_chart(fig, use_container_width=True)


with tab_outliers:
    if not num_cols:
        st.info("No numeric columns to inspect.")
    else:
        k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
        report = outliers_overview(df, k=k)
        st.dataframe(report, use_container_width=True)
        if not report.empty:
            top_outlier = report.iloc[0]["column"]
            col = st.selectbox(
                "Inspect column",
                num_cols,
                index=num_cols.index(top_outlier) if top_outlier in num_cols else 0,
            )
            fig = px.box(df, y=col, points="all", title=f"Outlier profile — {col}")
            st.plotly_chart(fig, use_container_width=True)


with tab_pairs:
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for pairwise plots.")
    else:
        chosen = st.multiselect(
            "Numeric columns to plot",
            num_cols,
            default=num_cols[: min(4, len(num_cols))],
        )
        color_options = ["—"] + cat_cols
        color = st.selectbox("Colour by category", color_options)
        if len(chosen) >= 2:
            color_arg = None if color == "—" else color
            fig = px.scatter_matrix(
                df, dimensions=chosen, color=color_arg,
                title="Scatter matrix",
                height=700,
            )
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            st.plotly_chart(fig, use_container_width=True)
