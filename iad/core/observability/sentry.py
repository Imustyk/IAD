"""Sentry SDK initialization — optional, env-driven.

When ``IAD_SENTRY_DSN`` is unset, all functions are no-ops so local dev and CI
never require a Sentry project.
"""
from __future__ import annotations

from typing import Any

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.observability.sentry")

_INITIALIZED = False


def is_sentry_enabled() -> bool:
    settings = get_settings()
    return bool(settings.SENTRY_DSN) and settings.SENTRY_ENABLED


def init_sentry(*, service: str = "iad") -> None:
    """Configure Sentry once per process if DSN is present."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    settings = get_settings()
    if not is_sentry_enabled():
        logger.debug("Sentry disabled (no DSN or SENTRY_ENABLED=false)")
        _INITIALIZED = True
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:  # pragma: no cover
        logger.warning("sentry-sdk not installed; pip install sentry-sdk")
        _INITIALIZED = True
        return

    integrations: list[Any] = [
        LoggingIntegration(level=None, event_level=None),
    ]

    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        integrations.extend([StarletteIntegration(), FastApiIntegration()])
    except ImportError:
        pass

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
        send_default_pii=False,
        integrations=integrations,
    )
    sentry_sdk.set_tag("service", service)
    logger.info("Sentry initialized", extra={"ctx_environment": settings.ENVIRONMENT})
    _INITIALIZED = True


def capture_exception(exc: BaseException, **context: Any) -> None:
    """Forward an exception to Sentry when enabled."""
    if not is_sentry_enabled():
        return
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in context.items():
                scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)
    except Exception as report_exc:  # pragma: no cover
        logger.debug("Sentry capture failed: %s", report_exc)
