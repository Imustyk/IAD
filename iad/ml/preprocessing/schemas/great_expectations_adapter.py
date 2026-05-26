"""Optional Great Expectations adapter.

Why an adapter rather than a hard dependency?
    Great Expectations brings ~50MB of transitive dependencies and is
    designed for orchestrated pipelines (Airflow, Dagster). The platform's
    primary path is Pandera; GE is offered for users who already run a GE
    Data Docs site and want consistency.

The adapter is enabled only when ``great_expectations`` can be imported. Use
:func:`is_available` to detect this at runtime; calling any other function
without GE installed raises ``ImportError`` with a clear remediation hint.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

from iad.core.logging import get_logger

logger = get_logger("iad.ml.preprocessing.ge_adapter")


def is_available() -> bool:
    """Return True if ``great_expectations`` is importable."""
    try:
        import great_expectations  # noqa: F401
        return True
    except ImportError:
        return False


def _require_ge() -> None:
    if not is_available():
        raise ImportError(
            "great_expectations is not installed. "
            "Install it with `pip install great-expectations` or use Pandera schemas."
        )


def schema_to_expectations(schema: pa.DataFrameSchema) -> list[dict[str, Any]]:
    """Translate a Pandera schema into a list of GE-style expectation configs.

    The output is the JSON-friendly representation used by GE's
    ``ExpectationSuite``. It is built without importing GE so the function
    works whether or not GE is installed; pass the returned list to
    :func:`build_expectation_suite` to attach it to a real GE suite.
    """
    expectations: list[dict[str, Any]] = []
    for name, column in schema.columns.items():
        expectations.append(
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": name},
            }
        )
        if not column.nullable:
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": name},
                }
            )
        for check in column.checks or []:
            spec = _check_to_expectation(name, check)
            if spec is not None:
                expectations.append(spec)
    return expectations


def _check_to_expectation(column: str, check: pa.Check) -> dict[str, Any] | None:
    """Best-effort conversion of a Pandera check to a GE expectation."""
    name = getattr(check, "name", "") or str(check)
    stats = getattr(check, "statistics", {}) or {}
    if "isin" in name and "allowed_values" in stats:
        return {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": column, "value_set": list(stats["allowed_values"])},
        }
    if "in_range" in name and "min_value" in stats and "max_value" in stats:
        return {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": column,
                "min_value": stats["min_value"],
                "max_value": stats["max_value"],
            },
        }
    if name in {"greater_than", "gt"} and "min_value" in stats:
        return {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": column,
                "min_value": stats["min_value"],
                "strict_min": True,
            },
        }
    if name in {"greater_than_or_equal_to", "ge"} and "min_value" in stats:
        return {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": column, "min_value": stats["min_value"]},
        }
    return None


def build_expectation_suite(
    schema: pa.DataFrameSchema, suite_name: str = "iad_suite"
) -> Any:  # pragma: no cover — only runs when GE is installed
    """Build a real GE ExpectationSuite from a Pandera schema."""
    _require_ge()
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    from great_expectations.core.expectation_suite import ExpectationSuite

    suite = ExpectationSuite(expectation_suite_name=suite_name)
    for spec in schema_to_expectations(schema):
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type=spec["expectation_type"],
                kwargs=spec["kwargs"],
            )
        )
    return suite


def validate_with_great_expectations(  # pragma: no cover — only runs when GE is installed
    df: pd.DataFrame, schema: pa.DataFrameSchema, suite_name: str = "iad_suite"
) -> dict[str, Any]:
    """Validate a DataFrame using GE built from a Pandera schema."""
    _require_ge()
    import great_expectations as ge

    suite = build_expectation_suite(schema, suite_name=suite_name)
    ge_df = ge.from_pandas(df)
    result = ge_df.validate(expectation_suite=suite)
    return result.to_json_dict()


__all__ = [
    "is_available",
    "schema_to_expectations",
    "build_expectation_suite",
    "validate_with_great_expectations",
]
