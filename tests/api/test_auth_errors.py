"""Auth API error-path tests."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_refresh_invalid_token(api_client) -> None:
    response = api_client.post("/auth/refresh", json={"refresh_token": "not-a-valid-token"})
    assert response.status_code == 401


@pytest.mark.integration
def test_change_password_wrong_current(api_client, auth_headers) -> None:
    response = api_client.post(
        "/auth/change-password",
        headers=auth_headers,
        json={"current_password": "WrongPassword", "new_password": "NewPassword456"},
    )
    assert response.status_code == 401
