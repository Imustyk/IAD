"""AuthService unit tests."""
from __future__ import annotations

import pytest

from iad.backend.services.auth_service import AuthService
from iad.core.exceptions import AuthError, ValidationError


def test_register_and_login(db_session) -> None:
    svc = AuthService()
    user, tokens = svc.register(
        email="alice@example.com",
        password="Password123",
        full_name="Alice",
        session=db_session,
    )
    assert user.email == "alice@example.com"
    assert tokens.access_token

    user2, tokens2 = svc.login(
        email="alice@example.com",
        password="Password123",
        session=db_session,
    )
    assert user2.id == user.id
    assert tokens2.refresh_token


def test_login_wrong_password(db_session) -> None:
    svc = AuthService()
    svc.register(email="bob@example.com", password="Password123", session=db_session)
    with pytest.raises(AuthError):
        svc.login(email="bob@example.com", password="WrongPassword1", session=db_session)


def test_duplicate_email(db_session) -> None:
    svc = AuthService()
    svc.register(email="dup@example.com", password="Password123", session=db_session)
    with pytest.raises(ValidationError):
        svc.register(email="dup@example.com", password="Password456", session=db_session)


def test_refresh_token(db_session) -> None:
    svc = AuthService()
    _, tokens = svc.register(email="refresh@example.com", password="Password123", session=db_session)
    new_tokens = svc.refresh(tokens.refresh_token, session=db_session)
    assert new_tokens.access_token
    assert new_tokens.refresh_token
