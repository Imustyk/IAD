"""JWT access and refresh tokens (python-jose)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from jose import JWTError, jwt

from iad.config.settings import Settings, get_settings
from iad.core.exceptions import AuthError

TokenType = Literal["access", "refresh"]


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims used by API dependencies."""

    sub: str  # user id
    email: str
    role: str
    token_type: TokenType
    exp: datetime
    jti: str | None = None

    @property
    def user_id(self) -> str:
        return self.sub


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0  # seconds until access token expiry


def _utcnow() -> datetime:
    return datetime.now(UTC)


def create_access_token(
    *,
    user_id: str,
    email: str,
    role: str,
    settings: Settings | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)``."""
    cfg = settings or get_settings()
    expires_delta = timedelta(minutes=cfg.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    return _create_token(
        user_id=user_id,
        email=email,
        role=role,
        token_type="access",
        expires_delta=expires_delta,
        settings=cfg,
        extra_claims=extra_claims,
    )


def create_refresh_token(
    *,
    user_id: str,
    email: str,
    role: str,
    settings: Settings | None = None,
) -> str:
    cfg = settings or get_settings()
    expires_delta = timedelta(days=cfg.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    token, _ = _create_token(
        user_id=user_id,
        email=email,
        role=role,
        token_type="refresh",
        expires_delta=expires_delta,
        settings=cfg,
        extra_claims={"jti": str(uuid.uuid4())},
    )
    return token


def _create_token(
    *,
    user_id: str,
    email: str,
    role: str,
    token_type: TokenType,
    expires_delta: timedelta,
    settings: Settings,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, int]:
    now = _utcnow()
    expire = now + expires_delta
    claims: dict[str, Any] = {
        "sub": user_id,
        "email": email.lower(),
        "role": role,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),
    }
    if extra_claims:
        claims.update(extra_claims)
    encoded = jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded, int(expires_delta.total_seconds())


def decode_token(token: str, *, expected_type: TokenType | None = None, settings: Settings | None = None) -> TokenPayload:
    """Decode and validate a JWT; raise :class:`AuthError` on failure."""
    cfg = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            cfg.SECRET_KEY,
            algorithms=[cfg.JWT_ALGORITHM],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise AuthError(
            "Invalid or expired token.",
            user_message="Your session has expired. Please sign in again.",
        ) from exc

    token_type = payload.get("type")
    if expected_type and token_type != expected_type:
        raise AuthError(
            f"Expected token type {expected_type}, got {token_type}",
            user_message="Invalid authentication token.",
        )

    sub = payload.get("sub")
    email = payload.get("email")
    role = payload.get("role", "viewer")
    exp_ts = payload.get("exp")
    if not sub or not email or exp_ts is None:
        raise AuthError("Token missing required claims.", user_message="Invalid authentication token.")

    return TokenPayload(
        sub=str(sub),
        email=str(email),
        role=str(role),
        token_type=token_type,  # type: ignore[arg-type]
        exp=datetime.fromtimestamp(int(exp_ts), tz=UTC),
        jti=payload.get("jti"),
    )
