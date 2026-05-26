"""Logging configuration — handlers attached, files written, namespacing."""
from __future__ import annotations

import logging

from iad.config.settings import get_settings
from iad.core.logging import configure_logging, get_logger


def test_get_logger_returns_namespaced_logger() -> None:
    logger = get_logger("frontend.home")
    assert isinstance(logger, logging.Logger)
    assert logger.name.startswith("iad.")


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    before = len(logging.getLogger().handlers)
    configure_logging()
    after = len(logging.getLogger().handlers)
    assert before == after


def test_configure_logging_force_reattaches_handlers() -> None:
    configure_logging()
    configure_logging(force=True)
    handlers = logging.getLogger().handlers
    assert any(
        isinstance(h, logging.handlers.RotatingFileHandler)  # type: ignore[attr-defined]
        for h in handlers
    )


def test_log_files_are_written_on_emit(tmp_path) -> None:
    configure_logging()
    logger = get_logger("iad.test.logs")
    logger.error("emit me to the file")
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:  # pragma: no cover
            pass
    settings = get_settings()
    assert (settings.LOGS_DIR / "app.log").exists()
    assert (settings.LOGS_DIR / "errors.log").exists()


def test_training_logger_has_dedicated_file_handler() -> None:
    configure_logging(force=True)
    training = logging.getLogger("iad.training")
    has_file = any(
        isinstance(h, logging.handlers.RotatingFileHandler)  # type: ignore[attr-defined]
        for h in training.handlers
    )
    assert has_file
