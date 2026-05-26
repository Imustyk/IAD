"""Authentication use cases — register, login, token refresh."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from iad.backend.database.session import session_scope
from iad.backend.models.user import User
from iad.backend.repositories.unit_of_work import UnitOfWork
from iad.backend.schemas.auth import TokenResponse, UserResponse
from iad.backend.security.csrf import generate_csrf_token
from iad.backend.security.jwt_tokens import (
    TokenPair,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from iad.backend.security.passwords import hash_password, needs_rehash, verify_password
from iad.backend.security.permissions import Role
from iad.core.exceptions import AuthError, ValidationError
from iad.core.logging import get_logger

logger = get_logger("iad.auth")


@dataclass(frozen=True)
class AuthenticatedUser:
    """Domain object returned after successful authentication."""

    id: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    is_superuser: bool

    @classmethod
    def from_orm(cls, user: User) -> AuthenticatedUser:
        role = Role.from_user_flags(is_superuser=user.is_superuser, role=user.role)
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=role.value,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
        )

    def to_response(self) -> UserResponse:
        return UserResponse(
            id=self.id,
            email=self.email,
            full_name=self.full_name,
            role=self.role,
            is_active=self.is_active,
            is_superuser=self.is_superuser,
        )


class AuthService:
    """Register, authenticate and issue JWT token pairs."""

    def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None = None,
        session: Session | None = None,
    ) -> tuple[AuthenticatedUser, TokenPair]:
        """Create a new user account and return tokens."""

        def _run(sess: Session) -> tuple[AuthenticatedUser, TokenPair]:
            uow = UnitOfWork(sess)
            existing = uow.users.get_by_email(email)
            if existing is not None:
                raise ValidationError(
                    "Email already registered.",
                    user_message="An account with this email already exists.",
                    field="email",
                )
            user = uow.users.create(
                email=email,
                full_name=full_name,
                hashed_password=hash_password(password),
                role=Role.ANALYST.value,
            )
            auth_user = AuthenticatedUser.from_orm(user)
            tokens = self._issue_tokens(auth_user)
            logger.info("user registered", extra={"user_id": user.id, "email": user.email})
            return auth_user, tokens

        if session is not None:
            return _run(session)
        with session_scope() as sess:
            return _run(sess)

    def login(
        self,
        *,
        email: str,
        password: str,
        session: Session | None = None,
    ) -> tuple[AuthenticatedUser, TokenPair]:
        """Verify credentials and return tokens."""

        def _run(sess: Session) -> tuple[AuthenticatedUser, TokenPair]:
            uow = UnitOfWork(sess)
            user = uow.users.get_by_email(email)
            if user is None or not verify_password(password, user.hashed_password):
                raise AuthError(
                    "Invalid email or password.",
                    user_message="Invalid email or password.",
                )
            if not user.is_active:
                raise AuthError(
                    "Account is disabled.",
                    user_message="This account has been deactivated.",
                )
            if user.hashed_password and needs_rehash(user.hashed_password):
                user.hashed_password = hash_password(password)
                sess.flush()
            auth_user = AuthenticatedUser.from_orm(user)
            tokens = self._issue_tokens(auth_user)
            logger.info("user login", extra={"user_id": user.id})
            return auth_user, tokens

        if session is not None:
            return _run(session)
        with session_scope() as sess:
            return _run(sess)

    def refresh(self, refresh_token: str, session: Session | None = None) -> TokenPair:
        """Issue a new access token from a valid refresh token."""
        payload = decode_token(refresh_token, expected_type="refresh")

        def _run(sess: Session) -> TokenPair:
            uow = UnitOfWork(sess)
            user = uow.users.get_or_raise(payload.user_id)
            if not user.is_active:
                raise AuthError("Account is disabled.", user_message="This account has been deactivated.")
            auth_user = AuthenticatedUser.from_orm(user)
            return self._issue_tokens(auth_user)

        if session is not None:
            return _run(session)
        with session_scope() as sess:
            return _run(sess)

    def get_user_by_id(self, user_id: str, session: Session | None = None) -> AuthenticatedUser:
        def _run(sess: Session) -> AuthenticatedUser:
            uow = UnitOfWork(sess)
            user = uow.users.get_or_raise(user_id)
            if not user.is_active:
                raise AuthError("Account is disabled.")
            return AuthenticatedUser.from_orm(user)

        if session is not None:
            return _run(session)
        with session_scope() as sess:
            return _run(sess)

    def change_password(
        self,
        user_id: str,
        *,
        current_password: str,
        new_password: str,
        session: Session | None = None,
    ) -> None:
        def _run(sess: Session) -> None:
            uow = UnitOfWork(sess)
            user = uow.users.get_or_raise(user_id)
            if not verify_password(current_password, user.hashed_password):
                raise AuthError("Current password is incorrect.", user_message="Current password is incorrect.")
            uow.users.update_password(user_id, hash_password(new_password))

        if session is not None:
            _run(session)
        else:
            with session_scope() as sess:
                _run(sess)

    @staticmethod
    def _issue_tokens(user: AuthenticatedUser) -> TokenPair:
        access, expires_in = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )
        refresh = create_refresh_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )
        return TokenPair(access_token=access, refresh_token=refresh, expires_in=expires_in)

    @staticmethod
    def to_token_response(tokens: TokenPair, *, include_csrf: bool = True) -> TokenResponse:
        csrf = generate_csrf_token() if include_csrf else None
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            csrf_token=csrf,
        )
