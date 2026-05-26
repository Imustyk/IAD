"""Prometheus middleware tests."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iad.backend.middleware.prometheus import PrometheusMiddleware


def test_prometheus_middleware_records_request() -> None:
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware)

    @app.get("/probe")
    def probe() -> dict[str, str]:
        return {"ok": "1"}

    client = TestClient(app)
    response = client.get("/probe")
    assert response.status_code == 200
