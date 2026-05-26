"""Home dashboard — overview and pipeline (navigation in sidebar only)."""
from __future__ import annotations

import streamlit as st

from iad.frontend.components.metric_cards import MetricSpec, render_metric_row
from iad.frontend.components.pipeline import render_pipeline_timeline
from iad.frontend.components.ui import render_hero, render_section_header
from iad.frontend.routes import DATA_LOADING
from iad.state.session import (
    KEY_BUSINESS_CASE,
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TRAINING_REPORT,
)


def render_kpi_row() -> None:
    df = st.session_state.get(KEY_DATASET)
    report = st.session_state.get(KEY_TRAINING_REPORT)
    bundle = st.session_state.get(KEY_MODEL_BUNDLE)

    specs: list[MetricSpec] = []

    if df is not None:
        name = st.session_state.get(KEY_DATASET_NAME) or "Dataset"
        missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
        specs.extend([
            MetricSpec("Rows", f"{df.shape[0]:,}"),
            MetricSpec("Columns", f"{df.shape[1]:,}"),
            MetricSpec(
                "Missing",
                f"{missing_pct:.1f}%",
                delta=f"{int(df.isna().sum().sum()):,} cells",
                delta_direction="negative" if missing_pct > 5 else "neutral",
            ),
            MetricSpec("Dataset", name[:32]),
        ])
    else:
        specs.append(MetricSpec("Dataset", "Not loaded", delta="Open Data in sidebar"))

    if report is not None:
        best = getattr(report, "best_model_name", None) or getattr(report, "best_entry", None)
        if hasattr(best, "model_name"):
            best = best.model_name
        task = getattr(report, "task_type", None) or getattr(report, "task", "—")
        specs.append(
            MetricSpec(
                "Best model",
                str(best)[:24] if best else "—",
                delta=str(task),
                delta_direction="positive",
            )
        )
    elif bundle is not None:
        specs.append(MetricSpec("Model", "Ready", delta="Apply model", delta_direction="positive"))
    else:
        specs.append(MetricSpec("Model", "Not trained", delta="Predictive analytics"))

    render_metric_row(specs[:4], columns=min(4, len(specs)))


def render_getting_started() -> None:
    if st.session_state.get(KEY_DATASET) is not None:
        return
    st.info("No dataset loaded. Open **Data → Data loading** in the sidebar.", icon=":material/info:")
    try:
        st.page_link(DATA_LOADING, label="Go to data loading", icon=":material/arrow_forward:")
    except Exception:
        pass


def render_business_case_form() -> None:
    case = st.session_state[KEY_BUSINESS_CASE]
    with st.form("business_case_form", clear_on_submit=False):
        case["title"] = st.text_input("Project title", value=case.get("title", ""))
        case["problem"] = st.text_area("Problem statement", value=case.get("problem", ""), height=72)
        case["objective"] = st.text_area("Objective", value=case.get("objective", ""), height=72)
        case["kpis"] = st.text_input("Key KPIs", value=case.get("kpis", ""))
        if st.form_submit_button("Save business case", type="primary", use_container_width=True):
            st.session_state[KEY_BUSINESS_CASE] = case
            st.success("Business case saved.")


def render_home_dashboard() -> None:
    render_hero(subtitle="Workspace overview. Use the sidebar to open each analytics stage.")
    st.markdown("")  # spacing between iframe panels
    render_kpi_row()
    render_getting_started()

    render_section_header(
        "Analytics pipeline",
        "Eight stages from ingestion to export. Open each stage from the sidebar.",
    )
    render_pipeline_timeline()

    target = st.session_state.get(KEY_TARGET_COLUMN)
    if target:
        st.caption(f"Target column: `{target}`")

    with st.expander("Business case (optional)", expanded=False):
        render_business_case_form()
