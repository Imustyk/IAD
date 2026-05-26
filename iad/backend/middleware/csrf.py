"""CSRF protection for cookie-based sessions (optional header validation)."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from iad.backend.security.csrf import CSRF_HEADER, validate_csrf_token
from iad.config.settings import get_settings
from iad.core.exceptions import AuthError

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class CSRFMiddleware(BaseHTTPMiddleware):
    """Validate CSRF token on state-changing requests when enabled.

    Bearer-token clients (``Authorization: Bearer``) skip CSRF — standard for
    SPA/API clients. Cookie-based clients must send ``X-CSRF-Token``.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.CSRF_ENABLED:
            return await call_next(request)

        if request.method not in _UNSAFE_METHODS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return await call_next(request)

        if request.url.path.startswith(("/auth/", "/exports/", "/train", "/predict", "/upload")):
            return await call_next(request)

        token = request.headers.get(CSRF_HEADER)
        try:
            validate_csrf_token(token)
        except AuthError as exc:
            return JSONResponse(
                status_code=403,
                content={"code": exc.code, "message": exc.user_message},
            )

        return await call_next(request)
