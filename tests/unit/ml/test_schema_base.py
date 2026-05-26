"""Pandera schema adapter tests."""
from __future__ import annotations

import pandera.pandas as pa
import pytest

from iad.ml.preprocessing.exceptions import SchemaValidationFailed
from iad.ml.preprocessing.schemas.base import (
    SchemaValidationResult,
    coerce_with_schema,
    list_required_columns,
    schemas_describe,
    validate_with_pandera,
)


@pytest.fixture
def iris_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "sepal length (cm)": pa.Column(float, nullable=False),
            "species": pa.Column(str, nullable=False),
        },
        name="iris_subset",
    )


@pytest.mark.unit
def test_validate_success(iris_df, iris_schema) -> None:
    subset = iris_df.rename(
        columns={
            "sepal length (cm)": "sepal length (cm)",
            "species": "species",
        }
    )[["sepal length (cm)", "species"]]
    result = validate_with_pandera(subset, iris_schema)
    assert result.is_valid
    assert result.coerced_dataframe is not None


@pytest.mark.unit
def test_validate_failure_lazy(iris_df, iris_schema) -> None:
    bad = iris_df[["sepal length (cm)", "species"]].copy()
    bad.loc[0, "species"] = None
    result = validate_with_pandera(bad, iris_schema, lazy=True)
    assert not result.is_valid
    assert result.n_errors >= 1
    frame = result.to_frame()
    assert "column" in frame.columns


@pytest.mark.unit
def test_validate_raises(iris_df, iris_schema) -> None:
    bad = iris_df[["sepal length (cm)", "species"]].copy()
    bad["species"] = None
    with pytest.raises(SchemaValidationFailed):
        validate_with_pandera(bad, iris_schema, raise_on_error=True)


@pytest.mark.unit
def test_validate_non_dataframe_raises(iris_schema) -> None:
    with pytest.raises(SchemaValidationFailed):
        validate_with_pandera(None, iris_schema)  # type: ignore[arg-type]


@pytest.mark.unit
def test_coerce_with_schema(iris_df) -> None:
    out = coerce_with_schema(iris_df, {"sepal length (cm)": "float64"})
    assert out["sepal length (cm)"].dtype == "float64"


@pytest.mark.unit
def test_list_required_columns(iris_schema) -> None:
    cols = list_required_columns(iris_schema)
    assert "species" in cols


@pytest.mark.unit
def test_schemas_describe(iris_schema) -> None:
    table = schemas_describe([iris_schema])
    assert not table.empty


@pytest.mark.unit
def test_result_empty_to_frame() -> None:
    result = SchemaValidationResult(is_valid=True)
    assert result.to_frame().empty
