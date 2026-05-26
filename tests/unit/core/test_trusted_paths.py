"""Trusted artifact path resolution."""
from __future__ import annotations

import pytest

from iad.config.settings import get_settings
from iad.core.exceptions import ValidationError
from iad.core.paths import resolve_trusted_artifact_path


def test_resolve_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    get_settings.cache_clear()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("IAD_MODELS_DIR", str(models))
    get_settings.cache_clear()

    outside = tmp_path / "secret.joblib"
    outside.write_bytes(b"x")

    with pytest.raises(ValidationError, match="outside trusted"):
        resolve_trusted_artifact_path(outside)

    allowed = models / "ok.joblib"
    allowed.write_bytes(b"x")
    assert resolve_trusted_artifact_path(allowed) == allowed.resolve()

    get_settings.cache_clear()
