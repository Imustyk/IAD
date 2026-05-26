"""FastAPI dependencies — database session and current user."""
from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from iad.backend.database.session import get_session_factory
from iad.backend.security.jwt_tokens import TokenPayload, decode_token
from iad.backend.security.permissions import Permission, require_permission
from iad.backend.services.auth_service import AuthenticatedUser, AuthService
from iad.core.exceptions import AuthError, PermissionError_

_bearer = HTTPBearer(auto_error=False)


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]


def get_token_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenPayload:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials, expected_type="access")
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.user_message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


TokenDep = Annotated[TokenPayload, Depends(get_token_payload)]


def get_current_user(
    payload: TokenDep,
    db: DbSession,
) -> AuthenticatedUser:
    try:
        return AuthService().get_user_by_id(payload.user_id, session=db)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.user_message) from exc


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]


def require_perm(permission: Permission):
    """Factory for permission-checking dependencies."""

    def _checker(user: CurrentUser) -> AuthenticatedUser:
        try:
            require_permission(user.role, permission)
        except PermissionError_ as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.user_message) from exc
        return user

    return _checker
