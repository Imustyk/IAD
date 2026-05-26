"""Great Expectations adapter tests (Pandera translation path)."""
from __future__ import annotations

import pandera.pandas as pa
import pytest

from iad.ml.preprocessing.schemas.great_expectations_adapter import (
    is_available,
    schema_to_expectations,
)


@pytest.mark.unit
def test_is_available() -> None:
    assert isinstance(is_available(), bool)


@pytest.mark.unit
def test_schema_to_expectations() -> None:
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(int, nullable=False, checks=pa.Check.in_range(0, 10)),
            "y": pa.Column(str, nullable=True),
        },
        name="demo",
    )
    expectations = schema_to_expectations(schema)
    assert len(expectations) >= 2
    types = {e["expectation_type"] for e in expectations}
    assert "expect_column_to_exist" in types
