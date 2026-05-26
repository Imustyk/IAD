"""Prometheus metric definitions and helpers.

Uses a dedicated :class:`CollectorRegistry` so tests do not pollute the global
default registry and metric registration stays idempotent per process.
"""
from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.observability.prometheus")

_REGISTRY: CollectorRegistry = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS_TOTAL: Counter = Counter(
    "iad_http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "path", "status"],
    registry=_REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS: Histogram = Histogram(
    "iad_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_REGISTRY,
)

ML_OPERATIONS_TOTAL: Counter = Counter(
    "iad_ml_operations_total",
    "ML operations executed (train, predict, export, etc.)",
    ["operation", "outcome"],
    registry=_REGISTRY,
)

ML_OPERATION_DURATION_SECONDS: Histogram = Histogram(
    "iad_ml_operation_duration_seconds",
    "ML operation duration in seconds",
    ["operation"],
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0),
    registry=_REGISTRY,
)

TRAINING_JOBS_TOTAL: Counter = Counter(
    "iad_training_jobs_total",
    "Training jobs started or completed",
    ["outcome"],
    registry=_REGISTRY,
)

_APP_INFO_REGISTERED = False


def _ensure_app_info() -> None:
    """Register build info once (idempotent — safe across repeated test app boots)."""
    global _APP_INFO_REGISTERED
    if _APP_INFO_REGISTERED:
        return
    from prometheus_client import Info

    settings = get_settings()
    info = Info("iad_app", "IAD application build metadata", registry=_REGISTRY)
    info.info(
        {
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "app": settings.APP_NAME,
        }
    )
    _APP_INFO_REGISTERED = True


def metrics_enabled() -> bool:
    settings = get_settings()
    return settings.METRICS_ENABLED


def normalize_path(path: str) -> str:
    """Collapse dynamic segments to keep Prometheus cardinality bounded."""
    if path in ("/health", "/healthz", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"):
        return path
    parts = path.strip("/").split("/")
    normalized: list[str] = []
    for part in parts:
        if part.isdigit() or len(part) == 32 and all(c in "0123456789abcdef" for c in part.lower()):
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized) if normalized else "/"


def observe_http_request(*, method: str, path: str, status: int, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    _ensure_app_info()
    endpoint = normalize_path(path)
    HTTP_REQUESTS_TOTAL.labels(method=method.upper(), path=endpoint, status=str(status)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method.upper(), path=endpoint).observe(duration_seconds)


def observe_ml_operation(*, operation: str, outcome: str, duration_seconds: float | None = None) -> None:
    if not metrics_enabled():
        return
    _ensure_app_info()
    ML_OPERATIONS_TOTAL.labels(operation=operation, outcome=outcome).inc()
    if duration_seconds is not None:
        ML_OPERATION_DURATION_SECONDS.labels(operation=operation).observe(duration_seconds)


def inc_training_job(*, outcome: str) -> None:
    if not metrics_enabled():
        return
    _ensure_app_info()
    TRAINING_JOBS_TOTAL.labels(outcome=outcome).inc()


class RequestTimer:
    """Context manager recording HTTP latency into Prometheus."""

    __slots__ = ("_method", "_path", "_start")

    def __init__(self, method: str, path: str) -> None:
        self._method = method
        self._path = path
        self._start = 0.0

    def __enter__(self) -> RequestTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        duration = time.perf_counter() - self._start
        status = 500 if exc_type else 200
        observe_http_request(
            method=self._method,
            path=self._path,
            status=status,
            duration_seconds=duration,
        )


def generate_metrics() -> bytes:
    """Serialize all registered metrics for the ``/metrics`` scrape endpoint."""
    if not metrics_enabled():
        return b""
    _ensure_app_info()
    return generate_latest(_REGISTRY)


def content_type_latest() -> str:
    return CONTENT_TYPE_LATEST
