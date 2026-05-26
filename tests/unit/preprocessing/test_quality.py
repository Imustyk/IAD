"""Quality module — duplicates, nulls, outliers, impossible-value rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from iad.ml.preprocessing import (
    ImpossibleValueRule,
    check_impossible_values,
    columns_above_null_threshold,
    detect_duplicates,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    null_report,
)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def test_detect_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    rep = detect_duplicates(df)
    assert rep.n_duplicate_rows == 2  # two identical rows are both flagged
    assert rep.total_rows == 4


def test_detect_constant_columns() -> None:
    df = pd.DataFrame({"const": [1, 1, 1], "vary": [1, 2, 3]})
    rep = detect_duplicates(df)
    assert "const" in rep.constant_columns


def test_detect_duplicate_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [4, 5, 6]})
    rep = detect_duplicates(df)
    assert rep.n_duplicate_column_groups >= 1
    flat = [c for grp in rep.duplicate_column_groups for c in grp]
    assert "a" in flat and "b" in flat


def test_near_constant_columns_flagged() -> None:
    df = pd.DataFrame({"x": ["A"] * 99 + ["B"]})
    rep = detect_duplicates(df, near_constant_threshold=0.95)
    assert "x" in rep.near_constant_columns


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------
def test_null_report_basic() -> None:
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    rep = null_report(df, threshold=0.4)
    frame = rep.to_frame()
    assert frame.loc[frame["column"] == "a", "n_missing"].item() == 2
    assert frame.loc[frame["column"] == "a", "above_threshold"].item()
    assert "a" in rep.columns_above_threshold()


def test_columns_above_null_threshold_helper() -> None:
    df = pd.DataFrame({"x": [None] * 10, "y": list(range(10))})
    cols = columns_above_null_threshold(df, threshold=0.5)
    assert cols == ["x"]


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def test_iqr_outlier_detection() -> None:
    df = pd.DataFrame({"x": list(range(100)) + [1000]})
    rep = detect_outliers_iqr(df)
    assert rep.method == "iqr"
    assert rep.n_outliers[0] >= 1


def test_zscore_outlier_detection() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=200)
    df = pd.DataFrame({"x": np.concatenate([base, [50.0]])})
    rep = detect_outliers_zscore(df, threshold=3.0)
    assert rep.method == "zscore"
    assert rep.n_outliers[0] >= 1


def test_isolation_forest_runs() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    rep = detect_outliers_isolation_forest(df, contamination=0.05)
    assert rep.method == "isolation_forest"
    # Should report some outliers given contamination=0.05.
    assert rep.n_outliers[0] > 0


# ---------------------------------------------------------------------------
# Impossible-value rules
# ---------------------------------------------------------------------------
def test_range_rule_catches_out_of_range() -> None:
    df = pd.DataFrame({"age": [25, -1, 200, 33]})
    rules = [ImpossibleValueRule("age", "range", {"min": 0, "max": 120})]
    reports = check_impossible_values(df, rules)
    assert reports[0].n_violations == 2


def test_regex_rule_catches_invalid_email() -> None:
    df = pd.DataFrame({"email": ["a@b.com", "no-at", "ok@example.org"]})
    rules = [ImpossibleValueRule("email", "regex", {"pattern": r".+@.+\..+"})]
    reports = check_impossible_values(df, rules)
    assert reports[0].n_violations == 1


def test_isin_and_not_in_rules() -> None:
    df = pd.DataFrame({"status": ["new", "renewed", "banned", "new"]})
    isin = check_impossible_values(
        df, [ImpossibleValueRule("status", "isin", {"values": ["new", "renewed"]})]
    )
    not_in = check_impossible_values(
        df, [ImpossibleValueRule("status", "not_in", {"values": ["banned"]})]
    )
    assert isin[0].n_violations == 1
    assert not_in[0].n_violations == 1


def test_unknown_column_records_error() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    reports = check_impossible_values(
        df, [ImpossibleValueRule("missing", "non_negative")]
    )
    assert reports[0].error is not None


def test_max_length_rule() -> None:
    df = pd.DataFrame({"name": ["short", "way too long for the rule", "ok"]})
    reports = check_impossible_values(
        df, [ImpossibleValueRule("name", "max_length", {"n": 10})]
    )
    assert reports[0].n_violations == 1


def test_custom_rule() -> None:
    df = pd.DataFrame({"score": [0.5, 1.5, -0.1]})
    reports = check_impossible_values(
        df,
        [ImpossibleValueRule("score", "custom", {"predicate": lambda v: 0.0 <= v <= 1.0})],
    )
    assert reports[0].n_violations == 2
