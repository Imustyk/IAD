"""API middleware tests."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iad.backend.middleware.csrf import CSRFMiddleware
from iad.backend.middleware.rate_limit import RateLimitMiddleware
from iad.backend.middleware.security_headers import SecurityHeadersMiddleware
from iad.config.settings import get_settings


@pytest.fixture
def minimal_app(settings):
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/mutate")
    def mutate() -> dict[str, str]:
        return {"ok": "true"}

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(CSRFMiddleware)
    with TestClient(app) as client:
        yield client


@pytest.mark.unit
def test_security_headers_middleware(minimal_app) -> None:
    response = minimal_app.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.unit
def test_rate_limit_health_exempt(minimal_app) -> None:
    for _ in range(5):
        assert minimal_app.get("/health").status_code == 200


@pytest.mark.unit
def test_csrf_skipped_for_bearer(minimal_app, monkeypatch) -> None:
    monkeypatch.setenv("IAD_CSRF_ENABLED", "true")
    get_settings.cache_clear()
    response = minimal_app.post(
        "/mutate",
        headers={"Authorization": "Bearer fake-token"},
    )
    assert response.status_code == 200
    get_settings.cache_clear()
