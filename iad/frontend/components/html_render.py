"""Reliable HTML rendering without sandboxed iframes (avoids upload/session issues)."""
from __future__ import annotations

import html
from typing import Literal

import streamlit as st

PanelHeight = int | Literal["content"]


def esc(text: str) -> str:
    return html.escape(str(text))


def render_html_panel(
    body: str,
    *,
    height: PanelHeight = "content",
    scrolling: bool = False,
) -> None:
    """Render styled HTML in the main document (CSS injected once via theme.py)."""
    _ = height, scrolling  # kept for API compatibility; markdown auto-sizes
    st.markdown(body, unsafe_allow_html=True)


def render_compact_html(html_fragment: str) -> None:
    """Tiny inline HTML via markdown (sidebar snippets only)."""
    compact = " ".join(html_fragment.split())
    st.markdown(compact, unsafe_allow_html=True)
