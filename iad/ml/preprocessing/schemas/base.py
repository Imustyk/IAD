"""Pandera adapter — declarative schemas with structured failure reports.

The platform exposes Pandera as the primary schema engine because it is:
* lightweight (no Spark / Airflow dependencies),
* pandas-native,
* friendly with sklearn pipelines (validation can run before/after a fit),
* composable (schemas can be combined, lazily extended, and serialised).

Why a wrapper instead of using ``schema.validate`` directly?
    * Pandera raises ``SchemaErrors`` with a custom textual format. Our UI and
      API need a structured payload (``SchemaValidationResult``) with rows
      describing each violation. That is what ``validate_with_pandera``
      returns when ``raise_on_error=False``.
    * Pandera's exception types differ between the legacy and the pandas-only
      namespace; we localise the import here so the rest of the codebase is
      version-agnostic.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from iad.core.logging import get_logger
from iad.ml.preprocessing.exceptions import SchemaValidationFailed

logger = get_logger("iad.ml.preprocessing.schemas")


@dataclass(frozen=True)
class SchemaValidationResult:
    """Outcome of a schema validation.

    Attributes:
        is_valid: True when no errors were reported.
        errors: list of ``{column, check, failure_case, index}`` dicts.
        n_errors: convenience count.
        coerced_dataframe: the DataFrame as returned by Pandera (may have been
            type-coerced during validation if the schema sets ``coerce=True``).
    """

    is_valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    coerced_dataframe: pd.DataFrame | None = None

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    def to_frame(self) -> pd.DataFrame:
        """Render the violation list as a DataFrame for display."""
        if not self.errors:
            return pd.DataFrame(columns=["column", "check", "failure_case", "index"])
        return pd.DataFrame(self.errors)


def _failure_cases_to_records(failure_cases: pd.DataFrame | None) -> list[dict[str, Any]]:
    if failure_cases is None or failure_cases.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in failure_cases.iterrows():
        out.append(
            {
                "column": row.get("column"),
                "check": row.get("check"),
                "failure_case": row.get("failure_case"),
                "index": row.get("index"),
            }
        )
    return out


def validate_with_pandera(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    *,
    raise_on_error: bool = False,
    lazy: bool = True,
) -> SchemaValidationResult:
    """Validate a DataFrame against a Pandera schema.

    Args:
        df: the data to validate.
        schema: a ``pandera.pandas.DataFrameSchema``.
        raise_on_error: if True, raises ``SchemaValidationFailed`` on failure
            instead of returning a result with ``is_valid=False``.
        lazy: collect every error rather than failing on the first one.

    Returns:
        ``SchemaValidationResult`` with structured error information.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise SchemaValidationFailed("Expected a pandas DataFrame.", received_type=type(df).__name__)

    try:
        validated = schema.validate(df, lazy=lazy)
    except SchemaErrors as exc:
        errors = _failure_cases_to_records(getattr(exc, "failure_cases", None))
        logger.warning(
            "schema validation failed: %d errors", len(errors), extra={"ctx_n_errors": len(errors)}
        )
        if raise_on_error:
            raise SchemaValidationFailed(
                f"Schema validation failed with {len(errors)} error(s).",
                user_message="Dataset does not match the expected schema.",
                errors=errors[:10],  # cap context payload size
            ) from exc
        return SchemaValidationResult(is_valid=False, errors=errors)
    except SchemaError as exc:
        # Single-error path (when lazy=False or only one violation).
        record = {
            "column": getattr(exc, "schema", None) and getattr(exc.schema, "name", None),
            "check": str(getattr(exc, "check", "")),
            "failure_case": getattr(exc, "failure_cases", None),
            "index": None,
        }
        logger.warning("schema validation failed (single error)")
        if raise_on_error:
            raise SchemaValidationFailed(
                "Schema validation failed.",
                user_message="Dataset does not match the expected schema.",
                errors=[record],
            ) from exc
        return SchemaValidationResult(is_valid=False, errors=[record])

    return SchemaValidationResult(is_valid=True, coerced_dataframe=validated)


def coerce_with_schema(
    df: pd.DataFrame,
    column_dtypes: Mapping[str, str],
    *,
    drop_extra_columns: bool = False,
) -> pd.DataFrame:
    """Lightweight type coercion when a full Pandera schema is overkill.

    Useful when uploading raw CSVs and we want to ensure dtypes match before
    passing data to a fitted pipeline.
    """
    out = df.copy()
    for col, dtype in column_dtypes.items():
        if col not in out.columns:
            continue
        try:
            if dtype.startswith("datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce")
            else:
                out[col] = out[col].astype(dtype)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("could not coerce column %s to %s: %s", col, dtype, exc)
    if drop_extra_columns:
        out = out[[c for c in column_dtypes.keys() if c in out.columns]]
    return out


def list_required_columns(schema: pa.DataFrameSchema) -> list[str]:
    """Return the names of columns marked as ``required=True`` in the schema."""
    required: list[str] = []
    for name, column in schema.columns.items():
        if getattr(column, "required", True):
            required.append(name)
    return required


def schemas_describe(schemas: Iterable[pa.DataFrameSchema]) -> pd.DataFrame:
    """Tabular description of a collection of schemas (debug helper)."""
    rows = []
    for schema in schemas:
        for name, col in schema.columns.items():
            rows.append(
                {
                    "schema": getattr(schema, "name", None) or schema.__class__.__name__,
                    "column": name,
                    "dtype": str(col.dtype),
                    "nullable": col.nullable,
                    "required": col.required,
                    "checks": [str(c) for c in (col.checks or [])],
                }
            )
    return pd.DataFrame(rows)
