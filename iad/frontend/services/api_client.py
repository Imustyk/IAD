"""HTTP client for the FastAPI backend — health checks and future API calls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from iad.config.settings import get_settings


@dataclass(frozen=True)
class BackendHealth:
    """Result of a lightweight ``GET /health`` probe."""

    ok: bool
    status: str
    version: str | None = None
    environment: str | None = None
    base_url: str = ""
    error: str | None = None


@dataclass(frozen=True)
class BackendReadiness:
    """Result of ``GET /healthz`` (database + observability)."""

    ok: bool
    status: str
    database_connected: bool | None = None
    error: str | None = None


class BackendClient:
    """Thin wrapper around httpx for IAD API endpoints."""

    def __init__(self, *, base_url: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.api_base_url()).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.API_TIMEOUT_SECONDS

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def health(self) -> BackendHealth:
        """Ping ``GET /health`` — does not require authentication."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._url("/health"))
                response.raise_for_status()
                data = response.json()
            return BackendHealth(
                ok=str(data.get("status", "")).lower() == "ok",
                status=str(data.get("status", "unknown")),
                version=str(data.get("version")) if data.get("version") else None,
                environment=str(data.get("environment")) if data.get("environment") else None,
                base_url=self.base_url,
            )
        except Exception as exc:
            return BackendHealth(
                ok=False,
                status="offline",
                base_url=self.base_url,
                error=str(exc),
            )

    def readiness(self) -> BackendReadiness:
        """Ping ``GET /healthz`` for database readiness."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._url("/healthz"))
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            db = data.get("database") or {}
            connected = db.get("connected") if isinstance(db, dict) else None
            status = str(data.get("status", "unknown"))
            return BackendReadiness(
                ok=status == "ok",
                status=status,
                database_connected=connected if isinstance(connected, bool) else None,
            )
        except Exception as exc:
            return BackendReadiness(ok=False, status="offline", error=str(exc))

    def docs_url(self) -> str:
        return f"{self.base_url}/docs"
