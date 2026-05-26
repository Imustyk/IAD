"""Health and readiness probes."""
from __future__ import annotations

import time

from fastapi import APIRouter

from iad.backend.database.session import check_connection
from iad.backend.services.persistence_service import PersistenceService
from iad.config.settings import get_settings
from iad.core.observability.health import observability_status
from iad.core.observability.sentry import is_sentry_enabled

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/live")
def liveness() -> dict[str, object]:
    """Liveness probe — process is up (no dependency checks)."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 2)}


@router.get("/healthz")
def healthz() -> dict[str, object]:
    """Readiness — includes database connectivity when enabled."""
    settings = get_settings()
    db_ok = check_connection(settings) if settings.DATABASE_ENABLED else True
    persistence = PersistenceService().health()
    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "database": {**persistence, "connected": db_ok},
        "observability": observability_status(),
        "sentry": is_sentry_enabled(),
    }
