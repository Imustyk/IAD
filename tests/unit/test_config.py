"""Settings package — defaults, validators, singleton semantics."""
from __future__ import annotations

import pytest

from iad.config.settings import Settings, get_settings


def test_defaults_are_sensible(settings: Settings) -> None:
    assert settings.APP_NAME
    assert settings.APP_VERSION
    assert settings.ENVIRONMENT in {"development", "staging", "production", "test"}
    assert settings.MAX_UPLOAD_MB > 0
    assert settings.MAX_ROWS > 0
    assert settings.MAX_COLUMNS > 0
    assert 0.05 <= settings.DEFAULT_TEST_SIZE <= 0.5
    assert 2 <= settings.DEFAULT_CV_FOLDS <= 20


def test_directories_are_created(settings: Settings) -> None:
    assert settings.LOGS_DIR.exists()
    assert settings.DATA_DIR.exists()
    assert settings.MODELS_DIR.exists()
    assert settings.REPORTS_DIR.exists()
    assert settings.EXPORTS_DIR.exists()


def test_singleton_caching() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_invalid_test_size_raises() -> None:
    with pytest.raises(ValueError):
        Settings(DEFAULT_TEST_SIZE=0.99)  # type: ignore[arg-type]


def test_invalid_cv_folds_raises() -> None:
    with pytest.raises(ValueError):
        Settings(DEFAULT_CV_FOLDS=1)  # type: ignore[arg-type]


def test_safe_dict_strips_secrets(settings: Settings) -> None:
    payload = settings.safe_dict()
    assert payload["SECRET_KEY"] in ("***", "change-me-in-production")  # default isn't stripped
    # When set to a non-empty value, secrets must be redacted:
    object.__setattr__(settings, "DATABASE_URL", "postgresql://u:p@h/db")
    assert settings.safe_dict()["DATABASE_URL"] == "***"
