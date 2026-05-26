"""Error handler — page_guard, safe_action, handle_error semantics."""
from __future__ import annotations

import logging

import pytest

from iad.core.error_handler import handle_error, page_guard, safe_action
from iad.core.exceptions import ValidationError


def test_handle_error_with_iad_error_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="iad.error_handler"):
        handle_error(ValidationError("nope", field="x"), user_facing=False)
    assert any("ValidationError" in r.getMessage() or "validation_error" in r.getMessage()
               for r in caplog.records) or any(r.levelname == "WARNING" for r in caplog.records)


def test_handle_error_with_unknown_logs_exception(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="iad.error_handler"):
        handle_error(RuntimeError("boom"), user_facing=False)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_page_guard_swallows_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING), page_guard("test-page"):
        raise ValidationError("oops")
    # No exception should escape — that's the whole point of page_guard.
    assert caplog.records


def test_safe_action_returns_none_on_failure() -> None:
    @safe_action
    def boom() -> int:
        raise RuntimeError("fail")

    assert boom() is None


def test_safe_action_returns_value_on_success() -> None:
    @safe_action
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42
