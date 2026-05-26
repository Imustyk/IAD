"""Security primitive unit tests."""
from __future__ import annotations

import pytest

from iad.backend.security.csrf import generate_csrf_token, validate_csrf_token
from iad.backend.security.jwt_tokens import create_access_token, decode_token
from iad.backend.security.passwords import hash_password, verify_password
from iad.backend.security.permissions import Permission, Role, has_permission, require_permission
from iad.backend.security.rate_limit import RateLimitConfig, RateLimiter
from iad.backend.security.upload_policy import validate_upload
from iad.core.exceptions import PermissionError_, RateLimitError, UploadError


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("MyPassword99")
    assert verify_password("MyPassword99", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_access_roundtrip(settings) -> None:
    token, _expires = create_access_token(
        user_id="user-1",
        email="test@example.com",
        role="analyst",
        settings=settings,
    )
    payload = decode_token(token, expected_type="access", settings=settings)
    assert payload.user_id == "user-1"
    assert payload.email == "test@example.com"
    assert payload.role == "analyst"


def test_csrf_token_validates() -> None:
    token = generate_csrf_token("session-abc")
    validate_csrf_token(token, "session-abc")


def test_rbac_analyst_can_train() -> None:
    assert has_permission(Role.ANALYST, Permission.MODEL_TRAIN)
    assert not has_permission(Role.VIEWER, Permission.MODEL_TRAIN)


def test_rbac_require_raises() -> None:
    with pytest.raises(PermissionError_):
        require_permission(Role.VIEWER, Permission.MODEL_TRAIN)


def test_rate_limiter_blocks() -> None:
    limiter = RateLimiter(RateLimitConfig(max_requests=2, window_seconds=60))
    limiter.check("client-1")
    limiter.check("client-1")
    with pytest.raises(RateLimitError):
        limiter.check("client-1")


def test_upload_policy_rejects_bad_extension(settings) -> None:
    with pytest.raises(UploadError):
        validate_upload(filename="malware.exe", size_bytes=100, settings=settings)
