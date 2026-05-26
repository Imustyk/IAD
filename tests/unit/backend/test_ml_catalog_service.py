"""ML catalog service unit tests."""
from __future__ import annotations

import pytest

from iad.backend.services.ml_catalog_service import MLCatalogService
from iad.core.exceptions import ValidationError


def test_list_metrics_requires_filter(db_session) -> None:
    with pytest.raises(ValidationError):
        MLCatalogService().list_metrics(user_id="user-1", session=db_session)
