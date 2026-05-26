"""Lightweight, dependency-free validators used throughout the application.

These guard rails exist so the rest of the code can assume well-formed inputs.
Heavy schema-level validation (Pandera, Great Expectations, drift detection)
is deferred to Phase 2 and lives in ``iad.ml.preprocessing``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from iad.config.settings import get_settings
from iad.core.exceptions import (
    SchemaError,
    UploadError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def validate_uploaded_file(name: str, size_bytes: int) -> None:
    """Validate the metadata of an uploaded file before reading it.

    Raises:
        UploadError: extension not allowed or file too large.
    """
    settings = get_settings()
    suffix = Path(name).suffix.lower()
    if suffix not in settings.ALLOWED_FILE_EXTENSIONS:
        raise UploadError(
            f"File extension {suffix!r} is not supported.",
            user_message=(
                f"File type '{suffix}' is not allowed. "
                f"Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
            ),
            received=suffix,
            allowed=list(settings.ALLOWED_FILE_EXTENSIONS),
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise UploadError(
            f"Upload too large: {size_bytes} > {max_bytes} bytes.",
            user_message=(
                f"File exceeds the {settings.MAX_UPLOAD_MB} MB platform limit."
            ),
            size_bytes=size_bytes,
            limit_bytes=max_bytes,
        )


# ---------------------------------------------------------------------------
# DataFrame structural validation
# ---------------------------------------------------------------------------
def validate_dataframe(
    df: pd.DataFrame,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
) -> None:
    """Structural sanity checks on a DataFrame.

    Catches the most common upload pathologies — empty file, single column,
    accidentally-loaded huge dataset, duplicate column names — early enough
    that downstream services can assume a well-formed frame.
    """
    settings = get_settings()

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValidationError(
            "Expected a pandas DataFrame.",
            received_type=type(df).__name__ if df is not None else "None",
        )

    rows, cols = df.shape
    if rows < min_rows:
        raise ValidationError(
            f"Dataset must have at least {min_rows} rows (got {rows}).",
            user_message=f"Dataset must contain at least {min_rows} row(s).",
            rows=rows,
        )
    if cols < min_cols:
        raise ValidationError(
            f"Dataset must have at least {min_cols} columns (got {cols}).",
            user_message=f"Dataset must contain at least {min_cols} column(s).",
            cols=cols,
        )
    if rows > settings.MAX_ROWS:
        raise ValidationError(
            f"Dataset has {rows:,} rows; platform limit is {settings.MAX_ROWS:,}.",
            user_message=(
                f"Dataset is too large ({rows:,} rows). "
                f"Limit is {settings.MAX_ROWS:,}."
            ),
            rows=rows,
            limit=settings.MAX_ROWS,
        )
    if cols > settings.MAX_COLUMNS:
        raise ValidationError(
            f"Dataset has {cols:,} columns; platform limit is {settings.MAX_COLUMNS:,}.",
            user_message=(
                f"Dataset has too many columns ({cols:,}). "
                f"Limit is {settings.MAX_COLUMNS:,}."
            ),
            cols=cols,
            limit=settings.MAX_COLUMNS,
        )
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        raise SchemaError(
            f"Duplicate column names: {dupes}.",
            user_message=f"The dataset contains duplicate column names: {dupes}.",
            duplicates=dupes,
        )


# ---------------------------------------------------------------------------
# Column-level validation
# ---------------------------------------------------------------------------
def validate_columns_present(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Required columns missing: {missing}.",
            user_message=f"The dataset is missing required columns: {missing}.",
            missing=missing,
        )


def validate_target_column(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise SchemaError(
            f"Target column {target!r} is not in the dataframe.",
            user_message=f"Target column '{target}' is not present in the dataset.",
            target=target,
        )
    series = df[target]
    if series.notna().sum() == 0:
        raise ValidationError(
            f"Target column {target!r} contains only missing values.",
            user_message=f"Target '{target}' has no non-null values; cannot train.",
        )
    if series.dropna().nunique() < 2:
        raise ValidationError(
            f"Target column {target!r} has fewer than 2 distinct values.",
            user_message=(
                f"Target '{target}' has fewer than 2 distinct non-null values; "
                f"a model cannot be trained."
            ),
        )


def validate_inference_payload(
    payload: pd.DataFrame, expected_features: Iterable[str]
) -> pd.DataFrame:
    """Align an inbound prediction payload with the expected feature schema.

    * Missing columns are added as NaN (the trained pipeline's imputer will
      fill them).
    * Extra columns are dropped silently — they cannot affect a fitted
      ``ColumnTransformer`` but we strip them anyway for cleanliness.
    * Returns a copy to avoid mutating caller-owned data.
    """
    if payload is None or not isinstance(payload, pd.DataFrame):
        raise ValidationError(
            "Expected a pandas DataFrame for inference.",
            received_type=type(payload).__name__ if payload is not None else "None",
        )
    expected = list(expected_features)
    if not expected:
        raise ValidationError("No expected feature columns supplied.")

    aligned = payload.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    return aligned[expected]


# ---------------------------------------------------------------------------
# Datatype validation
# ---------------------------------------------------------------------------
def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise SchemaError(f"Column {column!r} not in dataframe.", column=column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise SchemaError(
            f"Column {column!r} is not numeric (dtype={df[column].dtype}).",
            user_message=f"Column '{column}' must be numeric for this operation.",
            column=column,
            dtype=str(df[column].dtype),
        )


def validate_categorical_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise SchemaError(f"Column {column!r} not in dataframe.", column=column)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 50:
        raise SchemaError(
            f"Column {column!r} looks numeric (cardinality "
            f"{series.nunique()}). Categorical operations may be misleading.",
            user_message=f"Column '{column}' looks numeric, not categorical.",
            column=column,
            cardinality=int(series.nunique()),
        )
