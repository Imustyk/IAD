"""CSRF token generation and validation (double-submit pattern)."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

from iad.config.settings import get_settings
from iad.core.exceptions import AuthError

CSRF_HEADER: Final[str] = "X-CSRF-Token"
CSRF_COOKIE: Final[str] = "iad_csrf"


def generate_csrf_token(session_id: str | None = None) -> str:
    """Create an HMAC-signed CSRF token bound to optional session id."""
    settings = get_settings()
    nonce = secrets.token_urlsafe(32)
    payload = f"{session_id or 'anon'}:{nonce}"
    sig = hmac.new(
        settings.SECRET_KEY.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{sig}"


def validate_csrf_token(token: str | None, session_id: str | None = None) -> None:
    """Raise :class:`AuthError` when the token is missing or invalid."""
    if not token:
        raise AuthError(
            "CSRF token missing.",
            user_message="Security validation failed. Refresh the page and try again.",
        )
    parts = token.rsplit(":", 1)
    if len(parts) != 2:
        raise AuthError("Malformed CSRF token.", user_message="Security validation failed.")
    payload, provided_sig = parts
    if session_id and not payload.startswith(f"{session_id}:"):
        raise AuthError("CSRF session mismatch.", user_message="Security validation failed.")
    settings = get_settings()
    expected_sig = hmac.new(
        settings.SECRET_KEY.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise AuthError("Invalid CSRF signature.", user_message="Security validation failed.")
