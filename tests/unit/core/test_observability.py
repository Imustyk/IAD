"""Phase 12 observability unit tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from iad.backend.api.app import create_app
from iad.config.settings import get_settings
from iad.core.observability.prometheus import (
    generate_metrics,
    metrics_enabled,
    normalize_path,
    observe_http_request,
    observe_ml_operation,
)
from iad.core.observability.sentry import init_sentry, is_sentry_enabled


def test_normalize_path_collapses_ids() -> None:
    assert normalize_path("/users/123/models/456") == "/users/{id}/models/{id}"


def test_prometheus_observe_and_scrape() -> None:
    observe_http_request(method="GET", path="/health", status=200, duration_seconds=0.01)
    observe_ml_operation(operation="train", outcome="success", duration_seconds=1.0)
    payload = generate_metrics()
    assert metrics_enabled()
    assert b"iad_http_requests_total" in payload
    assert b"iad_ml_operations_total" in payload


def test_metrics_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "iad_http_requests_total" in response.text


def test_healthz_includes_observability() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert "observability" in body
    assert body["observability"]["metrics_enabled"] is True


def test_liveness_probe() -> None:
    client = TestClient(create_app())
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sentry_disabled_without_dsn(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("IAD_SENTRY_DSN", "")
    monkeypatch.setenv("IAD_ENVIRONMENT", "test")
    assert is_sentry_enabled() is False
    init_sentry(service="test")
    get_settings.cache_clear()


def test_metrics_disabled_returns_empty(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("IAD_METRICS_ENABLED", "false")
    from iad.core.observability.prometheus import generate_metrics

    assert generate_metrics() == b""
    get_settings.cache_clear()


def test_observe_duration_decorator() -> None:
    from iad.core.observability.performance import observe_duration

    @observe_duration("unit_test_op")
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(1) == 2


def test_init_observability() -> None:
    from iad.core.observability import init_observability

    init_observability(service="unit-test")


def test_observability_status_payload() -> None:
    from iad.core.observability.health import observability_status

    payload = observability_status()
    assert payload["metrics_enabled"] is True
    assert payload["version"]


def test_inc_training_job_metric() -> None:
    from iad.core.observability.prometheus import inc_training_job

    inc_training_job(outcome="completed")


def test_capture_exception_noop_when_disabled() -> None:
    from iad.core.observability.sentry import capture_exception

    capture_exception(RuntimeError("test"), source="unit")


def test_metrics_endpoint_disabled(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("IAD_METRICS_ENABLED", "false")
    client = TestClient(create_app())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"disabled" in response.content
    get_settings.cache_clear()
