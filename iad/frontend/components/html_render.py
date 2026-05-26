"""Reliable HTML rendering via iframe (avoids markdown escaping)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Literal

import streamlit as st

_EMBED_CSS = (Path(__file__).resolve().parents[1] / "styles" / "embed.css").read_text(encoding="utf-8")

PanelHeight = int | Literal["content"]


def esc(text: str) -> str:
    return html.escape(str(text))


def _render_iframe(html_doc: str, *, height: PanelHeight = "content", scrolling: bool = False) -> None:
    """Prefer ``st.iframe``; fall back to ``st.components.v1.html`` on older Streamlit."""
    try:
        st.iframe(html_doc, height=height, scrolling=scrolling)
    except (TypeError, AttributeError):
        import streamlit.components.v1 as components

        fallback_height = height if isinstance(height, int) else 180
        components.html(
            html_doc,
            height=fallback_height,
            scrolling=scrolling or height == "content",
        )


def render_html_panel(
    body: str,
    *,
    height: PanelHeight = "content",
    scrolling: bool = False,
) -> None:
    """Render a styled HTML fragment inside an isolated iframe panel."""
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{_EMBED_CSS}</style>
</head>
<body class="iad-panel">{body}</body>
</html>"""
    _render_iframe(doc, height=height, scrolling=scrolling)


def render_compact_html(html_fragment: str) -> None:
    """Tiny inline HTML via markdown (sidebar snippets only)."""
    compact = " ".join(html_fragment.split())
    st.markdown(compact, unsafe_allow_html=True)
