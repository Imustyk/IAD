"""Standard page layout — config, CSS, sidebar, headers."""
from __future__ import annotations

import streamlit as st

from iad.core.logging import get_logger
from iad.frontend.auth.gate import require_authentication
from iad.frontend.components.navbar import setup_sidebar
from iad.frontend.styles.theme import inject_css, page_config
from iad.state.session import init_session_state

logger = get_logger("iad.frontend")


def setup_page(
    title: str,
    *,
    icon: str = "📊",
    caption: str | None = None,
    layout: str = "wide",
) -> None:
    """One-call page bootstrap: config, session, CSS, sidebar, title.

    Must be the **first** Streamlit call on the page (before any other
    ``st.*`` widget). Replaces duplicated ``set_page_config`` +
    ``init_session_state`` blocks in legacy pages.
    """
    page_config(title, icon=icon, layout=layout)
    init_session_state()
    require_authentication()
    inject_css()
    setup_sidebar()

    st.markdown(f'<h1 class="iad-page-title">{icon} {title}</h1>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="iad-page-caption">{caption}</p>', unsafe_allow_html=True)

    logger.debug("page_render", extra={"page_title": title})


def section(title: str) -> None:
    """Render a section heading."""
    st.markdown(f'<h3 class="iad-section-title">{title}</h3>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="iad-divider"/>', unsafe_allow_html=True)
