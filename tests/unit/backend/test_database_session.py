"""Database engine and health checks."""
from __future__ import annotations

from iad.backend.database.session import check_connection, reset_engine
from iad.config.settings import get_settings


def test_resolved_database_url_is_sqlite_in_test(settings) -> None:
    assert settings.resolved_database_url().startswith("sqlite")


def test_check_connection(settings) -> None:
    reset_engine()
    get_settings.cache_clear()
    assert check_connection(settings) is True
