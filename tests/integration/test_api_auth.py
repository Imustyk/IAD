"""FastAPI authentication integration tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iad.backend.api.app import create_app
from iad.backend.database.init_db import create_all_tables
from iad.backend.database.session import reset_engine
from iad.config.settings import get_settings


@pytest.fixture
def api_client(settings):
    reset_engine()
    get_settings.cache_clear()
    create_all_tables(settings)
    app = create_app()
    with TestClient(app) as client:
        yield client
    reset_engine()
    get_settings.cache_clear()


def test_health(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_login_me(api_client) -> None:
    reg = api_client.post(
        "/auth/register",
        json={
            "email": "api@example.com",
            "password": "Password123",
            "full_name": "API User",
        },
    )
    assert reg.status_code == 201
    body = reg.json()
    assert "access_token" in body
    assert body.get("csrf_token")

    login = api_client.post(
        "/auth/login",
        json={"email": "api@example.com", "password": "Password123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = api_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "api@example.com"
    assert me.json()["role"] in ("analyst", "admin", "viewer")


def test_unauthorized_me(api_client) -> None:
    response = api_client.get("/auth/me")
    assert response.status_code == 401


def test_security_headers(api_client) -> None:
    response = api_client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
