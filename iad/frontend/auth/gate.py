"""Streamlit authentication gate — opt-in via ``IAD_AUTH_ENABLED``."""
from __future__ import annotations

from typing import Any

import streamlit as st

from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.state.session import KEY_USER, state_get, state_set

logger = get_logger("iad.frontend.auth")

KEY_AUTH_TOKENS = "auth_tokens"


def get_current_user() -> dict[str, Any] | None:
    """Return the logged-in user dict from session state, or None."""
    user = state_get(KEY_USER)
    return user if isinstance(user, dict) else None


def is_authenticated() -> bool:
    return get_current_user() is not None


def _render_login_form() -> None:
    settings = get_settings()
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            f"""
            <div class="iad-tw iad-auth-wrap p-6">
              <h3 class="text-xl font-bold text-gray-900">Sign in</h3>
              <p class="mt-2 text-sm text-gray-600">
                {settings.APP_NAME} — authentication required
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
                if submitted:
                    _handle_login(email, password)

        with tab_register:
            with st.form("register_form"):
                email = st.text_input("Email", key="reg_email")
                full_name = st.text_input("Full name", key="reg_name")
                password = st.text_input("Password", type="password", key="reg_pass")
                password2 = st.text_input("Confirm password", type="password", key="reg_pass2")
                submitted = st.form_submit_button("Create account", use_container_width=True)
                if submitted:
                    if password != password2:
                        st.error("Passwords do not match.")
                    else:
                        _handle_register(email, password, full_name)


def _handle_login(email: str, password: str) -> None:
    from iad.backend.database.init_db import create_all_tables
    from iad.backend.services.auth_service import AuthService
    from iad.core.exceptions import AuthError, ValidationError

    settings = get_settings()
    if settings.AUTO_CREATE_DB:
        create_all_tables(settings)

    try:
        user, tokens = AuthService().login(email=email, password=password)
        _store_session(user, tokens)
        st.success(f"Welcome, {user.full_name or user.email}!")
        st.rerun()
    except (AuthError, ValidationError) as exc:
        st.error(exc.user_message)


def _handle_register(email: str, password: str, full_name: str | None) -> None:
    from iad.backend.database.init_db import create_all_tables
    from iad.backend.services.auth_service import AuthService
    from iad.core.exceptions import AuthError, ValidationError

    settings = get_settings()
    if settings.AUTO_CREATE_DB:
        create_all_tables(settings)

    try:
        user, tokens = AuthService().register(
            email=email,
            password=password,
            full_name=full_name or None,
        )
        _store_session(user, tokens)
        st.success("Account created — you are now signed in.")
        st.rerun()
    except (AuthError, ValidationError) as exc:
        st.error(exc.user_message)


def _store_session(user: Any, tokens: Any) -> None:
    state_set(
        KEY_USER,
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_superuser": user.is_superuser,
        },
    )
    state_set(
        KEY_AUTH_TOKENS,
        {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "expires_in": tokens.expires_in,
        },
    )


def logout() -> None:
    state_set(KEY_USER, None)
    state_set(KEY_AUTH_TOKENS, None)
    st.rerun()


def require_authentication() -> None:
    """Block page render until the user signs in (when auth is enabled)."""
    settings = get_settings()
    if not settings.AUTH_ENABLED:
        return

    if not settings.DATABASE_ENABLED and not settings.AUTO_CREATE_DB:
        st.warning(
            "Authentication is enabled but the database is not. "
            "Set `IAD_DATABASE_ENABLED=true` or `IAD_AUTO_CREATE_DB=true`."
        )
        st.stop()

    if is_authenticated():
        user = get_current_user()
        assert user is not None
        with st.sidebar:
            st.caption(f"Signed in as **{user.get('email', '')}** ({user.get('role', '')})")
            if st.button("Sign out", use_container_width=True):
                logout()
        return

    _render_login_form()
    st.stop()
