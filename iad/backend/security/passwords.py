"""Password hashing via bcrypt (production-grade, no passlib dependency)."""
from __future__ import annotations

import bcrypt

from iad.core.exceptions import ValidationError

MIN_PASSWORD_LENGTH = 10


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage."""
    _validate_password_strength(plain_password)
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    """Return True when the plaintext matches the stored hash."""
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def _validate_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
            user_message=f"Use at least {MIN_PASSWORD_LENGTH} characters.",
            field="password",
        )
    if password.strip() != password:
        raise ValidationError(
            "Password must not have leading or trailing whitespace.",
            user_message="Password must not start or end with spaces.",
            field="password",
        )
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    if not (has_letter and has_digit):
        raise ValidationError(
            "Password must contain at least one letter and one digit.",
            user_message="Include both letters and numbers in your password.",
            field="password",
        )


def needs_rehash(_hashed_password: str) -> bool:
    """Rehash policy hook — extend when rotating cost factor."""
    return False
