"""Prometheus HTTP metrics middleware."""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from iad.core.observability.prometheus import metrics_enabled, observe_http_request


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for each HTTP call."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not metrics_enabled():
            return await call_next(request)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            duration = time.perf_counter() - start
            observe_http_request(
                method=request.method,
                path=request.url.path,
                status=status,
                duration_seconds=duration,
            )
