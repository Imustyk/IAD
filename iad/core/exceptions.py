"""Typed exception hierarchy used across UI, services, ML and API.

Every error is:

* **Categorised** — a stable ``code`` (snake_case) usable in metrics and alerts.
* **User-safe** — a ``user_message`` that can be rendered to end users.
* **Developer-friendly** — the underlying ``message`` carries technical detail.
* **Context-rich** — ``context`` (a free-form dict) is auto-attached to logs
  by the global error handler.

Concrete subclasses signal *what kind* of failure occurred. UI / API can do
``isinstance`` checks instead of brittle string matching.
"""
from __future__ import annotations

from typing import Any


class IADError(Exception):
    """Root of the application's exception hierarchy."""

    code: str = "iad_error"
    user_message: str = "An unexpected error occurred."
    http_status: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
        **context: Any,
    ) -> None:
        self.message = message or self.user_message
        if user_message is not None:
            self.user_message = user_message
        self.context: dict[str, Any] = dict(context)
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
        }

    def __repr__(self) -> str:  # pragma: no cover
        ctx = f", context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}{ctx})"


# ---------------------------------------------------------------------------
# Configuration / boot
# ---------------------------------------------------------------------------
class ConfigError(IADError):
    code = "config_error"
    user_message = "Application configuration is invalid."
    http_status = 500


# ---------------------------------------------------------------------------
# Validation family
# ---------------------------------------------------------------------------
class ValidationError(IADError):
    code = "validation_error"
    user_message = "The provided input failed validation."
    http_status = 422


class SchemaError(ValidationError):
    code = "schema_error"
    user_message = "The dataset schema is invalid for this operation."


class UploadError(ValidationError):
    code = "upload_error"
    user_message = "The uploaded file is invalid."


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
class DataLoadError(IADError):
    code = "data_load_error"
    user_message = "Failed to load the dataset."
    http_status = 400


class DatabaseError(IADError):
    code = "database_error"
    user_message = "A database operation failed."
    http_status = 500


class NotFoundError(IADError):
    code = "not_found"
    user_message = "The requested resource was not found."
    http_status = 404


# ---------------------------------------------------------------------------
# ML layer
# ---------------------------------------------------------------------------
class TrainingError(IADError):
    code = "training_error"
    user_message = "Model training failed."
    http_status = 500


class InferenceError(IADError):
    code = "inference_error"
    user_message = "Model inference failed."
    http_status = 500


class NotTrainedError(IADError):
    code = "not_trained"
    user_message = "No trained model is available for this operation."
    http_status = 409


class AnalyticsError(IADError):
    code = "analytics_error"
    user_message = "The analytics operation could not be completed."
    http_status = 422


# ---------------------------------------------------------------------------
# Auth / Security (used from Phase 7+)
# ---------------------------------------------------------------------------
class AuthError(IADError):
    code = "auth_error"
    user_message = "Authentication failed."
    http_status = 401


class RateLimitError(AuthError):
    code = "rate_limit_exceeded"
    user_message = "Too many requests. Please wait and try again."
    http_status = 429


class PermissionError_(IADError):  # noqa: N801 — name matches Python builtin intentionally
    code = "permission_denied"
    user_message = "You do not have permission to perform this action."
    http_status = 403


# ---------------------------------------------------------------------------
# Export / reporting (Phase 13)
# ---------------------------------------------------------------------------
class ExportError(IADError):
    code = "export_error"
    user_message = "Report export failed."
    http_status = 422


__all__ = [
    "IADError",
    "ConfigError",
    "ValidationError",
    "SchemaError",
    "UploadError",
    "DataLoadError",
    "DatabaseError",
    "NotFoundError",
    "TrainingError",
    "InferenceError",
    "NotTrainedError",
    "AnalyticsError",
    "ExportError",
    "AuthError",
    "RateLimitError",
    "PermissionError_",
]
