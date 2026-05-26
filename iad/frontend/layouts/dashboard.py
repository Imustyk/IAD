"""Home dashboard — KPI cards, workflow, recent activity."""
from __future__ import annotations

import streamlit as st

from iad.config.settings import get_settings
from iad.frontend.components.metric_cards import MetricSpec, render_metric_row
from iad.frontend.layouts.page import divider, section
from iad.state.session import (
    KEY_BUSINESS_CASE,
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TRAINING_REPORT,
)


def _workflow_steps_html() -> str:
    steps = [
        ("📥", "Data Loading", "Upload, fetch or pick a dataset"),
        ("📊", "Descriptive", "Stats, distributions, missing values"),
        ("🔍", "Diagnostic", "Correlations, tests, outliers"),
        ("🤖", "Predictive", "Train and benchmark ML models"),
        ("🎯", "Apply Model", "Score new rows or batches"),
        ("💡", "Prescriptive", "What-if and recommendations"),
        ("🧠", "Advanced", "NLP, forecasting, clustering"),
        ("📄", "Export", "PDF, DOCX, reports"),
    ]
    cards = []
    for icon, name, desc in steps:
        cards.append(
            f"""
            <div class="iad-workflow-step iad-animate-in">
              <div class="icon">{icon}</div>
              <div class="name">{name}</div>
              <div class="desc">{desc}</div>
            </div>
            """
        )
    return '<div class="iad-workflow-grid">' + "".join(cards) + "</div>"


def render_kpi_row() -> None:
    """Top-level KPI cards reflecting current workspace state."""
    df = st.session_state.get(KEY_DATASET)
    report = st.session_state.get(KEY_TRAINING_REPORT)
    bundle = st.session_state.get(KEY_MODEL_BUNDLE)

    specs: list[MetricSpec] = []

    if df is not None:
        name = st.session_state.get(KEY_DATASET_NAME) or "Dataset"
        missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
        specs.extend([
            MetricSpec("Rows", f"{df.shape[0]:,}", icon="📋"),
            MetricSpec("Columns", f"{df.shape[1]:,}", icon="📐"),
            MetricSpec(
                "Missing",
                f"{missing_pct:.1f}%",
                delta=f"{int(df.isna().sum().sum()):,} cells",
                delta_direction="negative" if missing_pct > 5 else "neutral",
                icon="⚠️",
            ),
            MetricSpec("Dataset", name[:24], icon="📊"),
        ])
    else:
        specs.append(MetricSpec("Dataset", "Not loaded", icon="📥", delta="Open Data Loading"))

    if report is not None:
        best = getattr(report, "best_model_name", None) or getattr(report, "best_entry", None)
        if hasattr(best, "model_name"):
            best = best.model_name
        task = getattr(report, "task_type", None) or getattr(report, "task", "—")
        specs.append(
            MetricSpec(
                "Best model",
                str(best)[:20] if best else "—",
                delta=str(task),
                delta_direction="positive",
                icon="🤖",
            )
        )
    elif bundle is not None:
        specs.append(MetricSpec("Model", "Trained", delta="Ready", delta_direction="positive", icon="🤖"))
    else:
        specs.append(MetricSpec("Model", "None", delta="Train on Predictive page", icon="🤖"))

    row = specs[:4] if len(specs) > 4 else specs
    render_metric_row(row, columns=len(row) if row else 4)


def render_workflow() -> None:
    section("Analytics workflow")
    st.markdown(_workflow_steps_html(), unsafe_allow_html=True)


def render_business_case_form() -> None:
    section("Business case")
    case = st.session_state[KEY_BUSINESS_CASE]
    with st.form("business_case_form", clear_on_submit=False):
        case["title"] = st.text_input(
            "Project title",
            value=case.get("title", ""),
            placeholder="e.g. Reduce subscriber churn for TelcoCo",
        )
        case["problem"] = st.text_area(
            "Problem statement",
            value=case.get("problem", ""),
            height=80,
        )
        case["objective"] = st.text_area(
            "Objective",
            value=case.get("objective", ""),
            height=80,
        )
        case["kpis"] = st.text_input("Key KPIs", value=case.get("kpis", ""))
        case["stakeholders"] = st.text_input("Stakeholders", value=case.get("stakeholders", ""))
        case["data_sources"] = st.text_area("Data sources", value=case.get("data_sources", ""), height=70)
        if st.form_submit_button("Save business case", use_container_width=True):
            st.session_state[KEY_BUSINESS_CASE] = case
            st.success("Business case saved.")


def render_hero() -> None:
    settings = get_settings()
    st.markdown(
        f"""
        <div class="iad-glass-card iad-animate-in" style="margin-bottom:1.5rem">
          <div style="font-size:1.1rem;color:var(--iad-text-secondary);line-height:1.6">
            <strong>{settings.APP_NAME}</strong> turns raw tabular data into deployed
            predictive models — descriptive → diagnostic → predictive → prescriptive,
            entirely in your browser.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_dashboard() -> None:
    """Full home page body (call from ``app.py`` after sidebar setup)."""
    render_hero()
    render_kpi_row()
    divider()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        render_workflow()
        with st.expander("What this platform does", expanded=False):
            st.markdown(
                """
                - **Loads** CSV, Excel, JSON, Parquet from upload, URL or samples
                - **Validates** schema and quality (Pandera, drift checks)
                - **Describes** distributions, missing values, time series
                - **Diagnoses** correlations, hypothesis tests, outliers
                - **Predicts** with sklearn, XGBoost, LightGBM, CatBoost + Optuna
                - **Explains** predictions with SHAP and LIME
                - **Applies** models to new data (single row or batch)
                - **Prescribes** actions via what-if scenarios
                """
            )
    with col_right:
        render_business_case_form()

    divider()
    target = st.session_state.get(KEY_TARGET_COLUMN)
    if target:
        st.caption(f"Current target column: `{target}`")

    with st.expander("Course requirements mapping", expanded=False):
        st.markdown(
            """
            | Requirement | Page |
            | --- | --- |
            | Business case | Home |
            | Data sources | Data Loading |
            | Descriptive analysis | Descriptive Analysis |
            | Diagnostic analysis | Diagnostic Analysis |
            | Predictive ML | Predictive Modeling |
            | Prescriptive (optional) | Prescriptive Analysis |
            | Model application | Apply Model |
            """
        )
