"""Reliable HTML rendering via ``st.components.v1.html`` (avoids markdown escaping)."""
from __future__ import annotations

import html
from pathlib import Path

import streamlit.components.v1 as components

_EMBED_CSS = (Path(__file__).resolve().parents[1] / "styles" / "embed.css").read_text(encoding="utf-8")


def esc(text: str) -> str:
    return html.escape(str(text))


def render_html_panel(body: str, *, height: int, scrolling: bool = False) -> None:
    """Render a styled HTML fragment inside an iframe panel."""
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{_EMBED_CSS}</style>
</head>
<body>{body}</body>
</html>"""
    components.html(doc, height=height, scrolling=scrolling)


def render_compact_html(html_fragment: str) -> None:
    """Single-line HTML via markdown (for tiny snippets only)."""
    import streamlit as st

    compact = " ".join(html_fragment.split())
    st.markdown(compact, unsafe_allow_html=True)
