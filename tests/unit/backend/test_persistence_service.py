"""Persistence service unit tests."""
from __future__ import annotations

import pytest

from iad.backend.services.persistence_service import PersistenceService


@pytest.mark.unit
def test_persistence_health(settings) -> None:
    svc = PersistenceService()
    health = svc.health()
    assert "enabled" in health
    assert "connected" in health
    assert health["url_scheme"].startswith("sqlite")


@pytest.mark.unit
def test_schema_and_checksum(iris_df) -> None:
    schema = PersistenceService._schema_from_dataframe(iris_df)
    assert schema["shape"][0] == len(iris_df)
    assert len(schema["columns"]) == len(iris_df.columns)
    checksum = PersistenceService._checksum(iris_df)
    assert len(checksum) == 64


@pytest.mark.integration
def test_register_dataset_record(iris_df, db_session) -> None:
    svc = PersistenceService()
    result = svc.register_dataset(iris_df, name="pytest-ds", source="unit-test")
    assert result.dataset_id
    assert result.version >= 1
    assert result.slug
