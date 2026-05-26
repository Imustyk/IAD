"""Preprocessing-specific exceptions, rooted in iad.core.exceptions.IADError."""
from __future__ import annotations

from iad.core.exceptions import IADError, ValidationError


class PreprocessingError(IADError):
    """Generic failure inside the data-engineering pipeline."""

    code = "preprocessing_error"
    user_message = "Data preprocessing failed."
    http_status = 500


class SchemaValidationFailed(ValidationError):
    """A Pandera (or compatible) schema reported validation errors."""

    code = "schema_validation_failed"
    user_message = "Dataset failed schema validation."


class DriftDetectionError(PreprocessingError):
    """Raised when a drift computation cannot be carried out (e.g. mismatched columns)."""

    code = "drift_detection_error"
    user_message = "Drift detection could not be completed."


class TransformerNotFittedError(PreprocessingError):
    """A transformer was used before its ``fit`` method ran."""

    code = "transformer_not_fitted"
    user_message = "A transformer was used before being fitted."


__all__ = [
    "PreprocessingError",
    "SchemaValidationFailed",
    "DriftDetectionError",
    "TransformerNotFittedError",
]
