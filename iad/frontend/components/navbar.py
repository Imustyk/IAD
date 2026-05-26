"""Sidebar — grouped navigation (single source; no duplicate dashboard links)."""
from __future__ import annotations

import streamlit as st

from iad.config.settings import get_settings
from iad.frontend.components.alerts import render_status_pill
from iad.frontend.routes import NAV_GROUPS
from iad.state.session import (
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
)


def render_sidebar_branding() -> None:
    settings = get_settings()
    st.sidebar.markdown(
        f"""
        <div class="iad-tw border-b border-gray-200 pb-4 mb-2">
          <p class="text-lg font-bold tracking-tight text-blue-600">{settings.APP_NAME}</p>
          <p class="mt-1 text-xs text-gray-500">v{settings.APP_VERSION} · {settings.ENVIRONMENT}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_status() -> None:
    st.sidebar.markdown('<p class="iad-nav-group-label">Workspace</p>', unsafe_allow_html=True)
    if st.session_state.get(KEY_DATASET) is not None:
        name = st.session_state.get(KEY_DATASET_NAME) or "Dataset"
        df = st.session_state[KEY_DATASET]
        shape = f"{df.shape[0]:,} x {df.shape[1]:,}" if hasattr(df, "shape") else ""
        render_status_pill(f"{name} ({shape})", "ok")
    else:
        render_status_pill("No dataset loaded", "warn")

    if st.session_state.get(KEY_MODEL_BUNDLE) is not None:
        render_status_pill("Model ready", "ok")
    else:
        render_status_pill("No trained model", "info")

    target = st.session_state.get(KEY_TARGET_COLUMN)
    if target:
        render_status_pill(f"Target: {target}", "info")


def render_sidebar_navigation() -> None:
    """Grouped nav with icons; active page highlighted by Streamlit."""
    for group in NAV_GROUPS:
        st.sidebar.markdown(
            f'<p class="iad-nav-group-label">{group.title}</p>',
            unsafe_allow_html=True,
        )
        for item in group.items:
            try:
                st.sidebar.page_link(
                    item.path,
                    label=item.label,
                    icon=item.icon,
                    use_container_width=True,
                )
            except TypeError:
                st.sidebar.page_link(item.path, label=item.label, use_container_width=True)
            except Exception:
                st.sidebar.caption(item.label)


def render_sidebar_settings() -> None:
    settings = get_settings()
    st.sidebar.markdown(
        '<p class="iad-nav-group-label" style="margin-top:1rem">Settings</p>',
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Application settings", expanded=False):
        st.markdown(f"**Environment:** {settings.ENVIRONMENT}")
        st.markdown(f"**Version:** {settings.APP_VERSION}")
        st.markdown(f"**Debug:** `{settings.DEBUG}`")
        st.caption("Collapse the sidebar using the control at the top edge.")


def setup_sidebar() -> None:
    render_sidebar_branding()
    render_workspace_status()
    render_sidebar_navigation()
    render_sidebar_settings()
