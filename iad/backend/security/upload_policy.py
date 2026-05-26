"""Centralised upload validation — size, extension, MIME hints."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from iad.config.settings import Settings, get_settings
from iad.core.exceptions import UploadError

# Extension → acceptable MIME prefixes (first segment)
_EXTENSION_MIME: dict[str, tuple[str, ...]] = {
    ".csv": ("text/", "application/csv", "application/octet-stream"),
    ".tsv": ("text/", "application/octet-stream"),
    ".txt": ("text/", "application/octet-stream"),
    ".xlsx": ("application/vnd.openxmlformats-officedocument", "application/octet-stream"),
    ".xls": ("application/vnd.ms-excel", "application/octet-stream"),
    ".json": ("application/json", "text/"),
    ".parquet": ("application/octet-stream", "application/x-parquet"),
    ".joblib": ("application/octet-stream",),
}


def validate_upload(
    *,
    filename: str,
    size_bytes: int | None,
    content_type: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Validate filename, size and optional MIME type before reading bytes."""
    cfg = settings or get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in cfg.ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(cfg.ALLOWED_FILE_EXTENSIONS)
        raise UploadError(
            f"Extension {ext!r} not allowed.",
            user_message=f"File type not allowed. Accepted: {allowed}",
            filename=filename,
        )

    if size_bytes is not None:
        max_bytes = cfg.MAX_UPLOAD_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise UploadError(
                f"File size {size_bytes} exceeds {max_bytes}.",
                user_message=f"File exceeds the {cfg.MAX_UPLOAD_MB} MB upload limit.",
                size_bytes=size_bytes,
            )
        if size_bytes == 0:
            raise UploadError(
                "Empty file uploaded.",
                user_message="The uploaded file is empty.",
            )

    if content_type and ext in _EXTENSION_MIME:
        allowed_prefixes = _EXTENSION_MIME[ext]
        if not any(content_type.startswith(p) for p in allowed_prefixes):
            raise UploadError(
                f"MIME {content_type!r} does not match extension {ext}.",
                user_message="File content type does not match the file extension.",
                content_type=content_type,
            )


def validate_model_upload(filename: str, size_bytes: int | None) -> None:
    """Stricter validation for ``.joblib`` model bundles."""
    validate_upload(filename=filename, size_bytes=size_bytes)
    if not filename.lower().endswith(".joblib"):
        raise UploadError(
            "Model uploads must be .joblib files.",
            user_message="Only .joblib model bundles are accepted.",
        )


def read_with_size_cap(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read stream up to *max_bytes*; raise if exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadError(
                f"Stream exceeded {max_bytes} bytes.",
                user_message="File is too large.",
            )
        chunks.append(chunk)
    return b"".join(chunks)
