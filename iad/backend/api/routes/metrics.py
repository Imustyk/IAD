"""Prometheus scrape endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Response

from iad.core.observability.prometheus import content_type_latest, generate_metrics, metrics_enabled

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in text exposition format."""
    if not metrics_enabled():
        return Response(content="# metrics disabled\n", media_type="text/plain")
    payload = generate_metrics()
    return Response(content=payload, media_type=content_type_latest())
