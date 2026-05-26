"""Prescriptive analytics page — what-if scenarios and recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from iad.frontend.components.charts import render_plotly
from iad.frontend.layouts.page import setup_page
from iad.frontend.streamlit_compat import dataframe
from src.prescriptive import (
    generate_recommendations,
    two_factor_scenario,
    what_if_scenario,
)
from src.utils import SESSION_KEYS, require_dataset


setup_page(
    "Prescriptive Analysis",
    caption="Step 6 (optional) — what-if scenarios and actionable recommendations.",
)

pipeline = st.session_state.get(SESSION_KEYS["model_bundle"])
report = st.session_state.get(SESSION_KEYS["training_report"])
features = st.session_state.get(SESSION_KEYS["feature_columns"]) or []
task_type = st.session_state.get(SESSION_KEYS["task_type"])

if pipeline is None or not features:
    st.warning(
        "Train a model on the **Predictive Modeling** page first to unlock "
        "scenario simulations."
    )
    st.stop()

df = require_dataset()
if df is None:
    st.stop()


st.subheader("Automatic recommendations")
for rec in generate_recommendations(report):
    st.markdown(f"- {rec}")


st.divider()
st.subheader("What-if simulator")

st.markdown(
    "Pick a baseline record, then sweep one or two features to see how the "
    "prediction would change while everything else is held constant."
)

baseline_mode = st.radio(
    "Baseline record",
    ["Mean / mode of dataset", "Median / mode of dataset", "Pick a row index"],
    horizontal=True,
)


def _baseline_row() -> pd.Series:
    available = df[[c for c in features if c in df.columns]]
    if baseline_mode == "Mean / mode of dataset":
        row = {}
        for col in features:
            if col in available.columns:
                series = available[col]
                if pd.api.types.is_numeric_dtype(series):
                    row[col] = float(series.mean(skipna=True)) if series.notna().any() else 0.0
                else:
                    mode = series.mode(dropna=True)
                    row[col] = mode.iloc[0] if not mode.empty else ""
            else:
                row[col] = np.nan
        return pd.Series(row)
    if baseline_mode == "Median / mode of dataset":
        row = {}
        for col in features:
            if col in available.columns:
                series = available[col]
                if pd.api.types.is_numeric_dtype(series):
                    row[col] = float(series.median(skipna=True)) if series.notna().any() else 0.0
                else:
                    mode = series.mode(dropna=True)
                    row[col] = mode.iloc[0] if not mode.empty else ""
            else:
                row[col] = np.nan
        return pd.Series(row)
    idx = st.number_input("Row index", 0, max(len(df) - 1, 0), 0, 1)
    return df.iloc[int(idx)][features].copy()


base_row = _baseline_row()
with st.expander("Baseline record"):
    dataframe(pd.DataFrame([base_row]))


numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_features:
    st.info("Need at least one numeric feature to run a sweep.")
    st.stop()


tab_one, tab_two = st.tabs(["One-feature sweep", "Two-feature heatmap"])

with tab_one:
    feature = st.selectbox("Feature to sweep", numeric_features)
    series = df[feature].dropna()
    lo = float(series.min())
    hi = float(series.max())
    span = st.slider("Sweep range", lo, hi, (lo, hi))
    steps = st.slider("Number of steps", 5, 80, 30, 5)
    grid = np.linspace(span[0], span[1], steps)

    if st.button("Run one-feature sweep", type="primary"):
        scenarios = what_if_scenario(pipeline, base_row, feature, grid, features, task_type)
        dataframe(scenarios.head(50))

        if task_type == "regression":
            fig = px.line(scenarios, x=feature, y="prediction",
                          title=f"Predicted {report.target} as {feature} varies")
            render_plotly(fig)
        else:
            proba_cols = [c for c in scenarios.columns if c.startswith("proba_")]
            if proba_cols:
                long = scenarios.melt(
                    id_vars=[feature],
                    value_vars=proba_cols,
                    var_name="class",
                    value_name="probability",
                )
                long["class"] = long["class"].str.replace("proba_", "")
                fig = px.line(long, x=feature, y="probability", color="class",
                              title=f"Class probabilities as {feature} varies")
                render_plotly(fig)
            else:
                fig = px.line(scenarios, x=feature, y="prediction",
                              title=f"Predicted class as {feature} varies")
                render_plotly(fig)


with tab_two:
    if len(numeric_features) < 2:
        st.info("Need at least two numeric features for a heatmap.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            feat_a = st.selectbox("Feature A", numeric_features, key="heat_a")
        with c2:
            feat_b = st.selectbox(
                "Feature B",
                [c for c in numeric_features if c != feat_a],
                key="heat_b",
            )
        steps = st.slider("Steps per axis", 5, 30, 15, 1, key="heat_steps")
        grid_a = np.linspace(df[feat_a].min(), df[feat_a].max(), steps)
        grid_b = np.linspace(df[feat_b].min(), df[feat_b].max(), steps)
        if st.button("Run two-feature sweep", type="primary"):
            scenarios = two_factor_scenario(
                pipeline, base_row, feat_a, grid_a, feat_b, grid_b, features, task_type,
            )
            if task_type == "regression":
                pivot = scenarios.pivot_table(
                    index=feat_b, columns=feat_a, values="prediction", aggfunc="mean",
                )
                fig = px.imshow(
                    pivot, aspect="auto",
                    color_continuous_scale="Viridis",
                    title=f"Predicted {report.target} surface",
                    labels=dict(color="prediction"),
                )
                render_plotly(fig)
            else:
                proba_cols = [c for c in scenarios.columns if c.startswith("proba_")]
                if proba_cols:
                    target_class = st.selectbox(
                        "Probability of class",
                        [c.replace("proba_", "") for c in proba_cols],
                    )
                    col = f"proba_{target_class}"
                    pivot = scenarios.pivot_table(
                        index=feat_b, columns=feat_a, values=col, aggfunc="mean",
                    )
                    fig = px.imshow(
                        pivot, aspect="auto",
                        color_continuous_scale="Viridis",
                        title=f"P({target_class}) surface",
                    )
                    render_plotly(fig)
                else:
                    dataframe(scenarios.head(200))
