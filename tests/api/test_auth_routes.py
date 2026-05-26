"""Extended auth API route tests."""
from __future__ import annotations

import uuid

import pytest


@pytest.mark.integration
def test_refresh_token(api_client) -> None:
    email = f"refresh-{uuid.uuid4().hex[:8]}@example.com"
    password = "Password123"
    reg = api_client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": "Refresh User"},
    )
    refresh = reg.json()["refresh_token"]
    response = api_client.post("/auth/refresh", json={"refresh_token": refresh})
    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.integration
def test_change_password(api_client, auth_headers) -> None:
    response = api_client.post(
        "/auth/change-password",
        headers=auth_headers,
        json={"current_password": "Password123", "new_password": "NewPassword456"},
    )
    assert response.status_code == 200


@pytest.mark.integration
def test_register_duplicate_email(api_client) -> None:
    email = f"dup-{uuid.uuid4().hex[:8]}@example.com"
    payload = {"email": email, "password": "Password123", "full_name": "Dup"}
    assert api_client.post("/auth/register", json=payload).status_code == 201
    dup = api_client.post("/auth/register", json=payload)
    assert dup.status_code == 422


@pytest.mark.integration
def test_login_invalid_password(api_client) -> None:
    email = f"badpw-{uuid.uuid4().hex[:8]}@example.com"
    api_client.post(
        "/auth/register",
        json={"email": email, "password": "Password123", "full_name": "User"},
    )
    response = api_client.post("/auth/login", json={"email": email, "password": "WrongPassword"})
    assert response.status_code == 401
