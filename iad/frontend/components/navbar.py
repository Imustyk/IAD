"""Sidebar navigation and workspace status."""
from __future__ import annotations

import streamlit as st

from iad.config.settings import get_settings
from iad.frontend.components.alerts import render_status_pill
from iad.frontend.styles.theme import theme_toggle_widget
from iad.state.session import (
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
)

# Multipage routes (must match files under pages/)
_NAV_PAGES: list[tuple[str, str, str]] = [
    ("app.py", "Home", "🏠"),
    ("pages/1_📥_Data_Loading.py", "Data Loading", "📥"),
    ("pages/2_📊_Descriptive_Analysis.py", "Descriptive", "📊"),
    ("pages/3_🔍_Diagnostic_Analysis.py", "Diagnostic", "🔍"),
    ("pages/4_🤖_Predictive_Modeling.py", "Predictive", "🤖"),
    ("pages/5_🎯_Apply_Model.py", "Apply Model", "🎯"),
    ("pages/6_💡_Prescriptive_Analysis.py", "Prescriptive", "💡"),
    ("pages/7_🧠_Advanced_Analytics.py", "Advanced", "🧠"),
    ("pages/8_📄_Export_Reports.py", "Export", "📄"),
]


def render_sidebar_branding() -> None:
    """App title and version in the sidebar."""
    settings = get_settings()
    st.sidebar.markdown(
        f"""
        <div style="padding:0.5rem 0 1rem 0">
          <div style="font-size:1.35rem;font-weight:700;color:var(--iad-primary)">
            {settings.APP_NAME}
          </div>
          <div style="font-size:0.8rem;color:var(--iad-text-muted)">
            v{settings.APP_VERSION} · {settings.ENVIRONMENT}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_status() -> None:
    """Show dataset / model / training status pills."""
    if st.session_state.get(KEY_DATASET) is not None:
        name = st.session_state.get(KEY_DATASET_NAME) or "Dataset"
        df = st.session_state[KEY_DATASET]
        shape = f"{df.shape[0]:,}×{df.shape[1]:,}" if hasattr(df, "shape") else ""
        render_status_pill(f"📊 {name} ({shape})", "ok")
    else:
        render_status_pill("No dataset loaded", "warn")

    if st.session_state.get(KEY_MODEL_BUNDLE) is not None:
        render_status_pill("🤖 Model ready", "ok")
    else:
        render_status_pill("No trained model", "info")

    target = st.session_state.get(KEY_TARGET_COLUMN)
    if target:
        render_status_pill(f"🎯 Target: {target}", "info")


def render_sidebar_navigation() -> None:
    """Clickable page links (Streamlit multipage API)."""
    st.sidebar.markdown('<div class="iad-nav-section">Navigate</div>', unsafe_allow_html=True)
    for page_path, label, icon in _NAV_PAGES:
        try:
            st.sidebar.page_link(page_path, label=f"{icon} {label}", use_container_width=True)
        except Exception:
            st.sidebar.caption(f"{icon} {label}")


def render_sidebar_extras() -> None:
    """Theme toggle and workflow hint."""
    st.sidebar.markdown("---")
    theme_toggle_widget()
    st.sidebar.markdown("---")
    st.sidebar.caption("Follow the workflow top-to-bottom for best results.")


def setup_sidebar() -> None:
    """Call from every page after ``init_session_state``."""
    render_sidebar_branding()
    render_workspace_status()
    render_sidebar_navigation()
    render_sidebar_extras()
