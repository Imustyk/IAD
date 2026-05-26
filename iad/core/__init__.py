"""Cross-cutting infrastructure: logging, exceptions, validation, paths.

Importing this package configures logging once (idempotent). Pages, services
and the CLI should all do::

    from iad.core import get_logger, page_guard
    logger = get_logger(__name__)
"""
from __future__ import annotations

from iad.core import validation
from iad.core.error_handler import handle_error, page_guard, safe_action
from iad.core.exceptions import (
    ConfigError,
    DataLoadError,
    IADError,
    InferenceError,
    NotTrainedError,
    SchemaError,
    TrainingError,
    UploadError,
    ValidationError,
)
from iad.core.logging import configure_logging, get_logger
from iad.core.paths import (
    data_dir,
    logs_dir,
    models_dir,
    project_root,
    safe_filename,
)

# Configure logging on first import. Idempotent (the function guards itself).
configure_logging()

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    # Errors / handlers
    "IADError",
    "ConfigError",
    "ValidationError",
    "SchemaError",
    "UploadError",
    "DataLoadError",
    "TrainingError",
    "InferenceError",
    "NotTrainedError",
    "handle_error",
    "page_guard",
    "safe_action",
    # Paths
    "project_root",
    "data_dir",
    "models_dir",
    "logs_dir",
    "safe_filename",
    # Modules
    "validation",
]
