"""Platform observability — metrics, error tracking, and performance hooks.

Phase 12 integrates:

* **Prometheus** — counters/histograms exposed at ``/metrics`` (API).
* **Sentry** — optional exception + performance tracing when ``IAD_SENTRY_DSN`` is set.
* **Health probes** — extended readiness in :mod:`iad.backend.api.routes.health`.

Call :func:`init_observability` once per process (API lifespan or Streamlit entry).
"""
from __future__ import annotations

from iad.core.observability.performance import observe_duration, timed_block
from iad.core.observability.prometheus import (
    content_type_latest,
    generate_metrics,
    inc_training_job,
    observe_http_request,
    observe_ml_operation,
)
from iad.core.observability.sentry import capture_exception, init_sentry, is_sentry_enabled

__all__ = [
    "capture_exception",
    "content_type_latest",
    "generate_metrics",
    "inc_training_job",
    "init_observability",
    "init_sentry",
    "is_sentry_enabled",
    "observe_duration",
    "observe_http_request",
    "observe_ml_operation",
    "timed_block",
]


def init_observability(*, service: str = "iad") -> None:
    """Initialize Sentry (if configured) and register process metadata."""
    init_sentry(service=service)
