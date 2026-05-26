"""Application-wide logging setup.

Design
------
* **Console** — RichHandler when available (pretty tracebacks in development),
  plain ``StreamHandler`` otherwise. Always at the configured level.
* **app.log** — rotating file handler at the configured level. Captures
  every meaningful event for post-mortem analysis.
* **errors.log** — rotating file handler at ERROR. Lets ops set up alerting
  on a small, dedicated stream without scanning the full app log.
* **training.log** — rotating file handler bound to the ``iad.training``
  logger. Lets ML engineers grep training history without noise from UI code.

Two formatters are supported:

* ``KeyValueFormatter`` — human-readable, used in development.
* ``JsonFormatter`` — line-delimited JSON for log aggregation backends
  (Loki / Datadog / ELK). Toggled via ``IAD_LOG_JSON=true``.

Idempotency
-----------
Streamlit re-runs scripts on every interaction; without a guard we would
attach handlers repeatedly and double-write each line. ``configure_logging()``
holds a module-level flag so it is safe to call as often as needed.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from typing import Any

from iad.config.settings import get_settings


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class KeyValueFormatter(logging.Formatter):
    """Single-line, human-friendly format with millisecond precision."""

    default_msec_format = "%s.%03d"

    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(module)s.%(funcName)s:%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record. ``ctx_*`` extras become top-level fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler builders
# ---------------------------------------------------------------------------
def _build_console_handler(level: int) -> logging.Handler:
    """Use Rich if importable, otherwise a plain StreamHandler."""
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=level,
            rich_tracebacks=True,
            markup=False,
            show_path=False,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(KeyValueFormatter())
        return handler


def _build_rotating_file_handler(
    filename: str, level: int, formatter: logging.Formatter
) -> logging.handlers.RotatingFileHandler:
    settings = get_settings()
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=settings.LOGS_DIR / filename,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CONFIGURED = False


def configure_logging(force: bool = False) -> None:
    """Install the application's logging stack.

    Args:
        force: re-attach handlers even if logging was already configured.
            Useful in tests when settings change between cases.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level_name = settings.LOG_LEVEL.upper()
    level = getattr(logging, level_name, logging.INFO)

    file_formatter: logging.Formatter
    file_formatter = JsonFormatter() if settings.LOG_JSON else KeyValueFormatter()

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.addHandler(_build_console_handler(level))
    root.addHandler(_build_rotating_file_handler("app.log", level, file_formatter))
    root.addHandler(_build_rotating_file_handler("errors.log", logging.ERROR, file_formatter))

    # Dedicated training channel — keeps ML logs separable from UI noise.
    training_logger = logging.getLogger("iad.training")
    for handler in list(training_logger.handlers):
        training_logger.removeHandler(handler)
    training_logger.addHandler(
        _build_rotating_file_handler("training.log", logging.INFO, file_formatter)
    )
    training_logger.propagate = True

    # Quiet down noisy third-party libraries.
    for noisy in (
        "matplotlib",
        "PIL",
        "fsspec",
        "asyncio",
        "urllib3",
        "watchdog",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Last-resort hook for unhandled exceptions in non-Streamlit contexts
    # (CLI, tests, scripts). Streamlit installs its own; this is harmless there.
    def _excepthook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("iad.unhandled").critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = _excepthook

    _CONFIGURED = True
    logging.getLogger("iad").info(
        "logging configured",
        extra={
            "ctx_environment": settings.ENVIRONMENT,
            "ctx_log_level": level_name,
            "ctx_json": settings.LOG_JSON,
            "ctx_logs_dir": str(settings.LOGS_DIR),
        },
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with logging configured. Always namespaced under ``iad.*``."""
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("iad"):
        name = f"iad.{name}" if name else "iad"
    return logging.getLogger(name)
