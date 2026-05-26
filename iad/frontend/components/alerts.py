"""User-facing alert and status components."""
from __future__ import annotations

import traceback
from typing import NoReturn

import streamlit as st

from iad.core.logging import get_logger

logger = get_logger(__name__)


def _message(text: str, prefix: str | None) -> str:
    if prefix:
        return f"{prefix} {text}"
    return text


def success(message: str, *, prefix: str | None = None) -> None:
    st.success(_message(message, prefix))


def warning(message: str, *, prefix: str | None = None) -> None:
    st.warning(_message(message, prefix))


def info(message: str, *, prefix: str | None = None) -> None:
    st.info(_message(message, prefix))


def error(
    message: str,
    *,
    prefix: str | None = None,
    show_details: bool = False,
    exc: Exception | None = None,
) -> None:
    st.error(_message(message, prefix))
    if exc is not None:
        logger.exception("UI error: %s", message, exc_info=exc)
    if show_details and exc is not None:
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def no_dataset_hint(page_name: str = "this page") -> None:
    info(f"No dataset loaded. Open **Data loading** first, then return to {page_name}.")


def render_status_pill(label: str, status: str) -> None:
    """Render a coloured status pill (ok / warn / info)."""
    css_class = status if status in ("ok", "warn", "info") else "info"
    st.markdown(
        f'<span class="iad-status-pill {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def guard_dataset_loaded(has_dataset: bool, page_name: str = "this page") -> bool:
    """Return False and show hint when no dataset is loaded."""
    if not has_dataset:
        no_dataset_hint(page_name)
        return False
    return True


def handle_page_error(exc: Exception, *, user_message: str = "An unexpected error occurred.") -> NoReturn:
    """Log, display, and stop the page."""
    error(user_message, show_details=True, exc=exc)
    st.stop()
