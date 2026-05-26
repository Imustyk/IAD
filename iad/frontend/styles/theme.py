"""Theme injection and dark-mode toggle for Streamlit.

Streamlit does not expose a first-class dark-mode API for custom CSS, so we
inject a ``<style>`` block on every rerun and drive the theme via a
``data-theme`` attribute on ``<html>`` and ``.stApp``. The toggle state is
persisted in ``st.session_state["iad_theme"]``.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from iad.config.settings import get_settings

THEME_KEY = "iad_theme"
DEFAULT_THEME = "light"  # fallback when settings are unavailable


def get_theme() -> str:
    """Return the active theme name ('light' or 'dark')."""
    if THEME_KEY not in st.session_state:
        settings = get_settings()
        default = settings.UI_DEFAULT_THEME
        if default not in ("light", "dark"):
            default = "light"
        st.session_state[THEME_KEY] = default
    return st.session_state[THEME_KEY]


def set_theme(theme: str) -> None:
    if theme not in ("light", "dark"):
        theme = "light"
    st.session_state[THEME_KEY] = theme


def _read_css_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _sync_theme_dom(theme: str) -> None:
    """Apply ``data-theme`` on ``<html>`` and Streamlit's root so token overrides work."""
    st.markdown(
        f"""
        <script>
        (function() {{
          const theme = {theme!r};
          document.documentElement.setAttribute("data-theme", theme);
          const app = window.parent.document.querySelector(".stApp")
            || document.querySelector(".stApp");
          if (app) app.setAttribute("data-theme", theme);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    """Inject the design-system CSS into the active Streamlit page."""
    style_dir = Path(__file__).parent
    tokens_css = _read_css_file(style_dir / "tokens.css")
    components_css = _read_css_file(style_dir / "components.css")

    theme = get_theme()
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
          background: var(--iad-bg) !important;
          color: var(--iad-text) !important;
          font-family: var(--iad-font-sans) !important;
        }}
        html[data-theme="{theme}"],
        .stApp[data-theme="{theme}"],
        [data-theme="{theme}"] {{
          color-scheme: {"dark" if theme == "dark" else "light"};
        }}
        {tokens_css}
        {components_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _sync_theme_dom(theme)


def theme_toggle_widget(key: str = "theme_toggle", label: str = "Dark mode") -> None:
    """Render a sidebar-friendly dark-mode toggle."""
    current = get_theme() == "dark"
    toggled = st.toggle(label, value=current, key=key)
    if toggled != current:
        set_theme("dark" if toggled else "light")
        st.rerun()


def page_config(
    title: str,
    icon: str = "📊",
    layout: str = "wide",
    *,
    inject_styles: bool = True,
) -> None:
    """Standardised ``st.set_page_config`` wrapper.

    ``st.set_page_config`` must be the first Streamlit command on a page;
    CSS injection runs immediately after.
    """
    settings = get_settings()
    st.set_page_config(
        page_title=f"{title} · {settings.APP_NAME}",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    if inject_styles:
        inject_css()
