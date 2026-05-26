"""Light-theme CSS injection for Streamlit shell and widgets."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from iad.config.settings import get_settings

THEME_KEY = "iad_theme"
DEFAULT_THEME = "light"


def get_theme() -> str:
    st.session_state[THEME_KEY] = "light"
    return "light"


def set_theme(_theme: str) -> None:
    st.session_state[THEME_KEY] = "light"


def _read_css_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _fonts_link() -> str:
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    """


def inject_css() -> None:
    style_dir = Path(__file__).parent
    tokens_css = _read_css_file(style_dir / "tokens.css")
    components_css = _read_css_file(style_dir / "components.css")
    bridge_css = _read_css_file(style_dir / "streamlit-bridge.css")

    st.markdown(_fonts_link(), unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
          font-family: 'Inter', var(--iad-font-sans), system-ui, sans-serif !important;
          background: #f9fafb !important;
          color: #111827 !important;
          color-scheme: light !important;
        }}
        .main .block-container {{
          max-width: var(--iad-content-max-width) !important;
          padding-top: 1.25rem !important;
          padding-bottom: 2.5rem !important;
        }}
        iframe[title="streamlit_components_v1.html"],
        iframe[title="streamlit.iframe"],
        [data-testid="stIFrame"] {{
          border: none !important;
          width: 100% !important;
          max-width: 100% !important;
          display: block !important;
          margin-bottom: 0.75rem !important;
          overflow: visible !important;
        }}
        [data-testid="stIFrame"] iframe {{
          display: block !important;
          width: 100% !important;
          min-height: 3rem !important;
          background: transparent !important;
        }}
        {tokens_css}
        {components_css}
        {bridge_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_config(
    title: str,
    icon: str | None = None,
    layout: str = "wide",
    *,
    inject_styles: bool = True,
) -> None:
    settings = get_settings()
    config: dict[str, object] = {
        "page_title": f"{title} · {settings.APP_NAME}",
        "layout": layout,
        "initial_sidebar_state": "expanded",
    }
    if icon:
        config["page_icon"] = icon
    st.set_page_config(**config)  # type: ignore[arg-type]
    if inject_styles:
        inject_css()
