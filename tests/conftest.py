"""Pytest configuration shared by every test module.

* Inserts the repository root into ``sys.path`` so both the new ``iad``
  package and the legacy ``src`` package are importable.
* Resets the cached ``Settings`` singleton between tests when fixtures
  change environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force the test environment so settings validators don't trip on dev defaults.
os.environ.setdefault("IAD_ENVIRONMENT", "test")
os.environ.setdefault("IAD_LOG_LEVEL", "WARNING")  # keep test output clean
os.environ.setdefault("IAD_PERF_USE_POLARS", "true")
os.environ.setdefault("IAD_PERF_USE_DASK", "true")


@pytest.fixture
def iris_df():
    from tests.helpers.factories import iris_dataframe

    return iris_dataframe()


@pytest.fixture
def regression_df():
    from tests.helpers.factories import regression_dataframe

    return regression_dataframe()


@pytest.fixture
def api_client(settings):
    """FastAPI TestClient with fresh in-memory database."""
    from fastapi.testclient import TestClient

    from iad.backend.api.app import create_app
    from iad.backend.database.init_db import create_all_tables
    from iad.backend.database.session import reset_engine
    from iad.config.settings import get_settings

    reset_engine()
    get_settings.cache_clear()
    create_all_tables(settings)
    app = create_app()
    with TestClient(app) as client:
        yield client
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def auth_headers(api_client) -> dict[str, str]:
    """Register a user and return Authorization header."""
    import uuid

    email = f"pytest-{uuid.uuid4().hex[:8]}@example.com"
    password = "Password123"
    reg = api_client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": "Pytest"},
    )
    assert reg.status_code == 201, reg.text
    login = api_client.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def settings():
    from iad.config.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    yield s
    get_settings.cache_clear()


@pytest.fixture
def db_session(settings):
    """In-memory SQLite session with fresh schema per test."""
    from iad.backend.database.init_db import create_all_tables, drop_all_tables
    from iad.backend.database.session import get_session_factory, reset_engine
    from iad.config.settings import get_settings

    reset_engine()
    get_settings.cache_clear()
    create_all_tables(settings)
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    finally:
        session.close()
        drop_all_tables(settings)
        reset_engine()
        get_settings.cache_clear()
