"""Upload policy validation tests."""
from __future__ import annotations

import io

import pytest

from iad.backend.security.upload_policy import (
    read_with_size_cap,
    validate_model_upload,
    validate_upload,
)
from iad.core.exceptions import UploadError


@pytest.mark.unit
def test_validate_upload_ok(settings) -> None:
    validate_upload(filename="data.csv", size_bytes=100, content_type="text/csv", settings=settings)


@pytest.mark.unit
def test_validate_upload_bad_extension(settings) -> None:
    with pytest.raises(UploadError):
        validate_upload(filename="virus.exe", size_bytes=10, settings=settings)


@pytest.mark.unit
def test_validate_upload_too_large(settings) -> None:
    huge = settings.MAX_UPLOAD_MB * 1024 * 1024 + 1
    with pytest.raises(UploadError):
        validate_upload(filename="big.csv", size_bytes=huge, settings=settings)


@pytest.mark.unit
def test_validate_upload_empty(settings) -> None:
    with pytest.raises(UploadError):
        validate_upload(filename="empty.csv", size_bytes=0, settings=settings)


@pytest.mark.unit
def test_validate_upload_mime_mismatch(settings) -> None:
    with pytest.raises(UploadError):
        validate_upload(
            filename="data.csv",
            size_bytes=10,
            content_type="application/pdf",
            settings=settings,
        )


@pytest.mark.unit
def test_validate_model_upload_rejects_non_joblib() -> None:
    validate_model_upload("model.joblib", 1024)
    with pytest.raises(UploadError):
        validate_model_upload("model.csv", 1024)


@pytest.mark.unit
def test_read_with_size_cap() -> None:
    data = read_with_size_cap(io.BytesIO(b"hello"), max_bytes=1024)
    assert data == b"hello"


@pytest.mark.unit
def test_read_with_size_cap_exceeded() -> None:
    with pytest.raises(UploadError):
        read_with_size_cap(io.BytesIO(b"x" * 20), max_bytes=10)
