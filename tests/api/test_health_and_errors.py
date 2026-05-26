"""API endpoint and error handler tests."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_healthz(api_client) -> None:
    for path in ("/health", "/healthz"):
        response = api_client.get(path)
        assert response.status_code == 200


@pytest.mark.integration
def test_validation_error(api_client) -> None:
    response = api_client.post("/auth/register", json={"email": "not-an-email"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"


@pytest.mark.integration
def test_auth_me_with_token(auth_headers, api_client) -> None:
    response = api_client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert "@" in response.json()["email"]
