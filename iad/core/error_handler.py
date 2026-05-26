"""Global error handling for Streamlit pages and callbacks.

Two complementary surfaces:

* :func:`page_guard` — context manager. Wrap an entire page body to catch and
  render any exception consistently.
* :func:`safe_action` — decorator. Wrap a callback (button / form handler).

Both rely on :func:`handle_error` which inspects the exception type and:

1. Logs it (warning for IADError, error for unknown exceptions, with full
   traceback).
2. Renders a user-safe message in the Streamlit UI.
3. In ``DEBUG`` builds, exposes the technical exception in a collapsible
   ``st.expander``.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from iad.config.settings import get_settings
from iad.core.exceptions import IADError
from iad.core.logging import get_logger
from iad.core.observability.sentry import capture_exception

logger = get_logger("iad.error_handler")

P = ParamSpec("P")
R = TypeVar("R")


def _streamlit_safe_render(message: str, *, exc: BaseException | None = None) -> None:
    """Render an error in Streamlit if available, otherwise no-op.

    Importing ``streamlit`` lazily so the module can be used from non-UI
    contexts (CLI, FastAPI, tests) without requiring a running Streamlit app.
    """
    try:
        import streamlit as st  # noqa: WPS433 (intentional local import)
    except Exception:  # pragma: no cover
        return
    try:
        st.error(message)
        if exc is not None and get_settings().DEBUG:
            with st.expander("Technical details (debug only)"):
                st.exception(exc)
    except Exception:  # pragma: no cover
        # Streamlit may not be running (e.g. inside a unit test that imported
        # the module). Fail silently — the log line above is still emitted.
        pass


def handle_error(exc: BaseException, *, user_facing: bool = True) -> None:
    """Central place to react to a thrown exception.

    Args:
        exc: the caught exception.
        user_facing: when True, render a Streamlit error message.
    """
    if isinstance(exc, IADError):
        logger.warning(
            "handled IADError code=%s message=%s",
            exc.code,
            exc.message,
            extra={"ctx_code": exc.code, "ctx_context": exc.context},
        )
        if user_facing:
            details = ""
            if exc.context:
                details = " (" + ", ".join(f"{k}={v}" for k, v in exc.context.items()) + ")"
            _streamlit_safe_render(f"❗ {exc.user_message}{details}", exc=exc)
        return

    logger.exception("unhandled exception: %s", exc, extra={"ctx_type": type(exc).__name__})
    capture_exception(exc, source="streamlit")
    if user_facing:
        _streamlit_safe_render(
            "💥 Unexpected error. The incident has been logged. "
            "Please retry or contact the platform team.",
            exc=exc,
        )


@contextmanager
def page_guard(page_name: str) -> Iterator[None]:
    """Wrap a Streamlit page body to log + render any error consistently.

    Usage::

        from iad.core import page_guard

        with page_guard("data_loading"):
            ...  # whole page body
    """
    started = time.perf_counter()
    logger.info("page enter: %s", page_name, extra={"ctx_page": page_name})
    try:
        yield
    except BaseException as exc:
        handle_error(exc)
    else:
        logger.debug(
            "page exit: %s (%.1f ms)",
            page_name,
            (time.perf_counter() - started) * 1000,
            extra={"ctx_page": page_name},
        )


def safe_action(func: Callable[P, R]) -> Callable[P, R | None]:
    """Decorator that catches and reports exceptions raised by a callback.

    Returns ``None`` on failure so callers can branch on a falsy result.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R | None:
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            handle_error(exc)
            return None

    return wrapper  # type: ignore[return-value]
