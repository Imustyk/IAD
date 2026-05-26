"""Path helpers — single source of truth for every filesystem location.

Always go through these functions instead of ``Path(__file__).parents[..]``
or hardcoded strings; that way settings / env overrides are honoured in tests
and in containerised deployments where directories are bind-mounted.
"""
from __future__ import annotations

from pathlib import Path

from iad.config.settings import get_settings
from iad.core.exceptions import ValidationError


def project_root() -> Path:
    return get_settings().PROJECT_ROOT


def data_dir() -> Path:
    return get_settings().DATA_DIR


def models_dir() -> Path:
    return get_settings().MODELS_DIR


def logs_dir() -> Path:
    return get_settings().LOGS_DIR


def reports_dir() -> Path:
    return get_settings().REPORTS_DIR


def exports_dir() -> Path:
    return get_settings().EXPORTS_DIR


def trusted_artifact_roots() -> list[Path]:
    """Directories from which model bundles may be loaded via API."""
    settings = get_settings()
    uploads = (settings.DATA_DIR / "uploads").resolve()
    roots = [
        settings.MODELS_DIR.resolve(),
        uploads,
        (uploads / "model_bundles").resolve(),
        settings.EXPORTS_DIR.resolve(),
    ]
    return [r for r in roots if r.exists() or r == settings.MODELS_DIR.resolve()]


def resolve_trusted_artifact_path(path: str | Path) -> Path:
    """Resolve *path* and ensure it lies under an allowed artifact directory.

    Prevents path-traversal attacks via ``artifact_path`` on ``POST /predict``.

    Raises:
        ValidationError: if the path escapes trusted roots or does not exist.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for root in trusted_artifact_roots():
        root_resolved = root.resolve()
        try:
            candidate.relative_to(root_resolved)
            if not candidate.is_file():
                raise ValidationError(
                    f"artifact not a file: {candidate}",
                    user_message="Model artifact file was not found.",
                )
            return candidate
        except ValueError:
            continue

    raise ValidationError(
        f"artifact path outside trusted directories: {candidate}",
        user_message="Model artifact path is not allowed. Use paths under models/ or data/uploads/.",
    )


_FILENAME_KEEP = set(
    "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def safe_filename(name: str, replacement: str = "_") -> str:
    """Sanitise a string for use as a filename — strips path separators and
    other unsafe characters, collapses leading/trailing replacement chars.
    """
    # Strip any directory components from user-supplied names.
    base = Path(name).name
    cleaned = "".join(c if c in _FILENAME_KEEP else replacement for c in base)
    cleaned = cleaned.strip(replacement)
    return cleaned or "file"
