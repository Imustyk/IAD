"""Impossible-value rule engine.

A small DSL for column-level invariants that the platform should never
encounter (negative ages, future timestamps, regex mismatches, ...). Rules
are declared as ``ImpossibleValueRule`` instances and evaluated by
:func:`check_impossible_values`. Each rule produces an
``ImpossibleValueReport`` describing how many rows violated it and a sample
of the violations for the UI.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from iad.core.logging import get_logger

logger = get_logger("iad.ml.preprocessing.quality.rules")


RuleType = Literal[
    "range",
    "non_negative",
    "non_zero",
    "regex",
    "isin",
    "not_in",
    "future_date",
    "past_date",
    "max_length",
    "custom",
]


@dataclass(frozen=True)
class ImpossibleValueRule:
    """Declarative rule.

    Examples:
        ``ImpossibleValueRule("age", "range", {"min": 0, "max": 120})``
        ``ImpossibleValueRule("email", "regex", {"pattern": r".+@.+\\..+"})``
        ``ImpossibleValueRule("created_at", "future_date", {})``
    """

    column: str
    rule_type: RuleType
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.rule_type not in {
            "range", "non_negative", "non_zero", "regex", "isin",
            "not_in", "future_date", "past_date", "max_length", "custom",
        }:
            raise ValueError(f"unknown rule_type: {self.rule_type}")


@dataclass(frozen=True)
class ImpossibleValueReport:
    """Report for a single rule evaluated on a single dataframe."""

    column: str
    rule_type: str
    description: str
    n_violations: int
    n_rows: int
    sample_violations: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def share_violations(self) -> float:
        return self.n_violations / max(self.n_rows, 1)


# ---------------------------------------------------------------------------
# Per-rule evaluators
# ---------------------------------------------------------------------------
def _eval_range(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    lo = params.get("min", -np.inf)
    hi = params.get("max", np.inf)
    return ~series.between(lo, hi)


def _eval_non_negative(series: pd.Series, _: dict[str, Any]) -> pd.Series:
    return series < 0


def _eval_non_zero(series: pd.Series, _: dict[str, Any]) -> pd.Series:
    return series == 0


def _eval_regex(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    pattern = params.get("pattern")
    if pattern is None:
        raise ValueError("regex rule requires a 'pattern' parameter")
    compiled = re.compile(pattern)
    return ~series.astype(str).str.match(compiled)


def _eval_isin(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    allowed = params.get("values")
    if allowed is None:
        raise ValueError("isin rule requires a 'values' parameter")
    return ~series.isin(allowed)


def _eval_not_in(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    forbidden = params.get("values")
    if forbidden is None:
        raise ValueError("not_in rule requires a 'values' parameter")
    return series.isin(forbidden)


def _eval_future_date(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    reference = params.get("now") or datetime.now()
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed > pd.Timestamp(reference)


def _eval_past_date(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    reference = params.get("now") or datetime.now()
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed < pd.Timestamp(reference)


def _eval_max_length(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    n = int(params.get("n", 0))
    return series.astype(str).str.len() > n


def _eval_custom(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    fn = params.get("predicate")
    if fn is None or not callable(fn):
        raise ValueError("custom rule requires a 'predicate' callable")
    return ~series.apply(fn).astype(bool)


_EVALUATORS = {
    "range": _eval_range,
    "non_negative": _eval_non_negative,
    "non_zero": _eval_non_zero,
    "regex": _eval_regex,
    "isin": _eval_isin,
    "not_in": _eval_not_in,
    "future_date": _eval_future_date,
    "past_date": _eval_past_date,
    "max_length": _eval_max_length,
    "custom": _eval_custom,
}


def check_impossible_values(
    df: pd.DataFrame, rules: Iterable[ImpossibleValueRule]
) -> list[ImpossibleValueReport]:
    """Evaluate every rule against ``df`` and return one report per rule."""
    results: list[ImpossibleValueReport] = []
    for rule in rules:
        if rule.column not in df.columns:
            results.append(
                ImpossibleValueReport(
                    column=rule.column,
                    rule_type=rule.rule_type,
                    description=rule.description,
                    n_violations=0,
                    n_rows=int(df.shape[0]),
                    error=f"column '{rule.column}' not in dataframe",
                )
            )
            continue
        series = df[rule.column]
        try:
            mask = _EVALUATORS[rule.rule_type](series, rule.params)
        except Exception as exc:
            logger.warning(
                "rule evaluation error for %s/%s: %s", rule.column, rule.rule_type, exc
            )
            results.append(
                ImpossibleValueReport(
                    column=rule.column,
                    rule_type=rule.rule_type,
                    description=rule.description,
                    n_violations=0,
                    n_rows=int(df.shape[0]),
                    error=str(exc),
                )
            )
            continue
        mask = mask.fillna(False)
        n_viol = int(mask.sum())
        sample = series[mask].head(10).tolist()
        results.append(
            ImpossibleValueReport(
                column=rule.column,
                rule_type=rule.rule_type,
                description=rule.description,
                n_violations=n_viol,
                n_rows=int(df.shape[0]),
                sample_violations=sample,
            )
        )
    return results
