"""Rate limiting middleware."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from iad.backend.security.rate_limit import get_rate_limiter
from iad.core.exceptions import RateLimitError


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply sliding-window rate limits per client IP."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in (
            "/health",
            "/healthz",
            "/live",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"ip:{client_ip}"

        try:
            get_rate_limiter().check(key, raise_on_limit=True)
        except RateLimitError as exc:
            return JSONResponse(
                status_code=exc.http_status,
                content={"code": exc.code, "message": exc.user_message},
            )

        return await call_next(request)
