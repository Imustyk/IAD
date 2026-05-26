"""Tests for the Streamlit → FastAPI client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iad.frontend.services.api_client import BackendClient


@pytest.mark.unit
def test_health_ok() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "version": "0.2.0",
        "environment": "development",
    }
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_response

    with patch("iad.frontend.services.api_client.httpx.Client", return_value=mock_client):
        result = BackendClient(base_url="http://127.0.0.1:8000").health()

    assert result.ok is True
    assert result.version == "0.2.0"
    assert result.base_url == "http://127.0.0.1:8000"


@pytest.mark.unit
def test_health_offline() -> None:
    with patch(
        "iad.frontend.services.api_client.httpx.Client",
        side_effect=ConnectionError("connection refused"),
    ):
        result = BackendClient(base_url="http://127.0.0.1:8000").health()

    assert result.ok is False
    assert result.status == "offline"
    assert "connection refused" in (result.error or "")
