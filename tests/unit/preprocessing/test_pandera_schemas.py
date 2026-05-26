"""Pandera schema validation."""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from iad.ml.preprocessing import (
    SAMPLE_SCHEMAS,
    SchemaValidationFailed,
    get_sample_schema,
    validate_with_pandera,
)


def test_get_sample_schema_returns_pandera_schema() -> None:
    schema = get_sample_schema("Iris (classification)")
    assert isinstance(schema, pa.DataFrameSchema)


def test_unknown_sample_schema_returns_none() -> None:
    assert get_sample_schema("UNKNOWN") is None


def test_iris_schema_validates_loaded_dataset() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    schema = SAMPLE_SCHEMAS["Iris (classification)"]
    result = validate_with_pandera(df, schema)
    assert result.is_valid, result.errors


def test_violation_returns_structured_errors() -> None:
    schema = pa.DataFrameSchema({"x": pa.Column(int, pa.Check.gt(0))}, strict=False, coerce=True)
    df = pd.DataFrame({"x": [-1, 2, -3, 4]})
    result = validate_with_pandera(df, schema)
    assert not result.is_valid
    assert result.n_errors >= 1


def test_validation_can_raise_on_error() -> None:
    schema = pa.DataFrameSchema({"x": pa.Column(int, pa.Check.gt(0))}, strict=False, coerce=True)
    df = pd.DataFrame({"x": [-1, -2]})
    with pytest.raises(SchemaValidationFailed):
        validate_with_pandera(df, schema, raise_on_error=True)


def test_telco_schema_validates_telco_sample() -> None:
    from src.data_loader import load_sample

    df = load_sample("Telco churn (classification)")
    schema = SAMPLE_SCHEMAS["Telco churn (classification)"]
    result = validate_with_pandera(df, schema)
    assert result.is_valid, result.errors


def test_ge_adapter_handles_missing_install() -> None:
    from iad.ml.preprocessing.schemas.great_expectations_adapter import (
        is_available,
        schema_to_expectations,
    )

    schema = SAMPLE_SCHEMAS["Iris (classification)"]
    expectations = schema_to_expectations(schema)
    assert isinstance(expectations, list)
    assert any(e["expectation_type"] == "expect_column_to_exist" for e in expectations)
    # is_available is a runtime check; assert it is a boolean.
    assert isinstance(is_available(), bool)
