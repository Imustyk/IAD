"""Standard page layout — config, CSS, sidebar, headers."""
from __future__ import annotations

import streamlit as st

from iad.core.logging import get_logger
from iad.frontend.auth.gate import require_authentication
from iad.frontend.components.navbar import setup_sidebar
from iad.frontend.components.ui import render_page_header
from iad.frontend.styles.theme import inject_css, page_config
from iad.state.session import init_session_state

logger = get_logger("iad.frontend")


def setup_page(
    title: str,
    *,
    icon: str | None = None,
    caption: str | None = None,
    layout: str = "wide",
) -> None:
    """One-call page bootstrap: config, session, CSS, sidebar, title."""
    page_config(title, icon=icon, layout=layout)
    init_session_state()
    require_authentication()
    inject_css()
    setup_sidebar()

    heading = f"{icon} {title}".strip() if icon else title
    render_page_header(heading, caption)

    logger.debug("page_render", extra={"page_title": title})


def section(title: str, description: str | None = None) -> None:
    """Render a section heading."""
    from iad.frontend.components.ui import render_section_header

    render_section_header(title, description)


def divider() -> None:
    st.markdown('<hr class="iad-divider"/>', unsafe_allow_html=True)
