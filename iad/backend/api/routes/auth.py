"""Authentication routes — register, login, refresh, profile."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from iad.backend.api.deps import CurrentUser, DbSession
from iad.backend.schemas.auth import (
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from iad.backend.services.auth_service import AuthService
from iad.core.exceptions import AuthError, ValidationError

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: DbSession) -> TokenResponse:
    try:
        _, tokens = AuthService().register(
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            session=db,
        )
        return AuthService.to_token_response(tokens)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.user_message) from exc


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    try:
        _, tokens = AuthService().login(email=body.email, password=body.password, session=db)
        return AuthService.to_token_response(tokens)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.user_message) from exc


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: DbSession) -> TokenResponse:
    try:
        tokens = AuthService().refresh(body.refresh_token, session=db)
        return AuthService.to_token_response(tokens)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.user_message) from exc


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser) -> UserResponse:
    return user.to_response()


@router.post("/change-password", response_model=MessageResponse)
def change_password(body: PasswordChangeRequest, user: CurrentUser, db: DbSession) -> MessageResponse:
    try:
        AuthService().change_password(
            user.id,
            current_password=body.current_password,
            new_password=body.new_password,
            session=db,
        )
        return MessageResponse(message="Password updated successfully.")
    except (AuthError, ValidationError) as exc:
        status_code = status.HTTP_401_UNAUTHORIZED if isinstance(exc, AuthError) else 422
        raise HTTPException(status_code=status_code, detail=exc.user_message) from exc
