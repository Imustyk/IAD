"""Validation layer — uploads, schemas, dtypes, inference payloads."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.core.exceptions import SchemaError, UploadError, ValidationError
from iad.core.validation import (
    validate_categorical_column,
    validate_columns_present,
    validate_dataframe,
    validate_inference_payload,
    validate_numeric_column,
    validate_target_column,
    validate_uploaded_file,
)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_valid_upload_passes() -> None:
    validate_uploaded_file("dataset.csv", 1024)
    validate_uploaded_file("Workbook.xlsx", 5_000_000)


def test_upload_rejects_unknown_extension() -> None:
    with pytest.raises(UploadError) as exc:
        validate_uploaded_file("malware.exe", 100)
    assert exc.value.code == "upload_error"
    assert ".exe" in str(exc.value.user_message).lower() or "allowed" in exc.value.user_message.lower()


def test_upload_rejects_oversized_file() -> None:
    too_big = 9_999_999_999_999  # 10 TB
    with pytest.raises(UploadError):
        validate_uploaded_file("data.csv", too_big)


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------
def test_valid_dataframe_passes() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validate_dataframe(df)


def test_empty_dataframe_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame())


def test_non_dataframe_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_dataframe("not a frame")  # type: ignore[arg-type]


def test_duplicate_columns_rejected() -> None:
    df = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(SchemaError):
        validate_dataframe(df)


# ---------------------------------------------------------------------------
# Column-level validation
# ---------------------------------------------------------------------------
def test_validate_columns_present_passes() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_columns_present(df, ["a", "b"])


def test_validate_columns_present_raises_on_missing() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(SchemaError) as exc:
        validate_columns_present(df, ["a", "missing"])
    assert exc.value.context["missing"] == ["missing"]


def test_validate_target_column_rejects_missing_or_constant() -> None:
    df_const = pd.DataFrame({"target": [1, 1, 1, 1]})
    with pytest.raises(ValidationError):
        validate_target_column(df_const, "target")

    with pytest.raises(SchemaError):
        validate_target_column(df_const, "no_such")


def test_validate_target_column_passes_on_valid() -> None:
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    validate_target_column(df, "target")


# ---------------------------------------------------------------------------
# Inference payload alignment
# ---------------------------------------------------------------------------
def test_inference_payload_aligns_columns() -> None:
    payload = pd.DataFrame({"age": [30], "extra": [99]})
    aligned = validate_inference_payload(payload, ["age", "income"])
    assert list(aligned.columns) == ["age", "income"]
    assert aligned["income"].isna().all()
    assert aligned.loc[0, "age"] == 30


def test_inference_payload_rejects_non_dataframe() -> None:
    with pytest.raises(ValidationError):
        validate_inference_payload([{"a": 1}], ["a"])  # type: ignore[arg-type]


def test_inference_payload_rejects_empty_features() -> None:
    with pytest.raises(ValidationError):
        validate_inference_payload(pd.DataFrame({"a": [1]}), [])


# ---------------------------------------------------------------------------
# Datatype validation
# ---------------------------------------------------------------------------
def test_validate_numeric_column() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    validate_numeric_column(df, "x")
    with pytest.raises(SchemaError):
        validate_numeric_column(df, "y")


def test_validate_categorical_column_warns_on_high_cardinality_numeric() -> None:
    df = pd.DataFrame({"hi_card": list(range(100))})
    with pytest.raises(SchemaError):
        validate_categorical_column(df, "hi_card")
