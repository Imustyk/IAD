"""Observability slice for health/readiness responses."""
from __future__ import annotations

from iad.config.settings import get_settings
from iad.core.observability.prometheus import metrics_enabled
from iad.core.observability.sentry import is_sentry_enabled


def observability_status() -> dict[str, object]:
    """Return a JSON-serializable observability summary for probes."""
    settings = get_settings()
    return {
        "metrics_enabled": metrics_enabled(),
        "metrics_path": settings.PROMETHEUS_METRICS_PATH if metrics_enabled() else None,
        "sentry_enabled": is_sentry_enabled(),
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
    }
