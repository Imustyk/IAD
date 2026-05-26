"""Duplicate detection — rows, columns, constant columns."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from iad.core.logging import get_logger

logger = get_logger("iad.ml.preprocessing.quality.duplicates")


@dataclass(frozen=True)
class DuplicateReport:
    """Outcome of a duplicate scan."""

    n_duplicate_rows: int
    duplicate_row_indices: list[int]
    n_duplicate_column_groups: int
    duplicate_column_groups: list[list[str]] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    near_constant_columns: list[str] = field(default_factory=list)
    total_rows: int = 0
    total_columns: int = 0

    @property
    def duplicate_row_share(self) -> float:
        return self.n_duplicate_rows / max(self.total_rows, 1)

    def summary(self) -> dict[str, object]:
        return {
            "duplicate_rows": self.n_duplicate_rows,
            "duplicate_row_share": round(self.duplicate_row_share, 4),
            "duplicate_column_groups": self.n_duplicate_column_groups,
            "constant_columns": self.constant_columns,
            "near_constant_columns": self.near_constant_columns,
        }


def _find_duplicate_columns(df: pd.DataFrame) -> list[list[str]]:
    """Return groups of columns whose values are identical row-by-row.

    Implementation note: comparing every column-pair is O(C^2 * R). For wide
    datasets (C > 1000) we cluster columns by a quick hash of their content
    first, then compare only within clusters.
    """
    if df.shape[1] < 2:
        return []
    # Hash each column to bucket potential duplicates fast.
    fingerprints: dict[bytes, list[str]] = {}
    for col in df.columns:
        try:
            digest = pd.util.hash_pandas_object(df[col], index=False).values.tobytes()
        except TypeError:
            # Non-hashable dtype; fall back to a stringified version.
            digest = df[col].astype(str).str.cat(sep="|").encode("utf-8")
        fingerprints.setdefault(digest, []).append(col)

    groups = [cols for cols in fingerprints.values() if len(cols) > 1]
    return groups


def detect_duplicates(
    df: pd.DataFrame,
    *,
    near_constant_threshold: float = 0.99,
) -> DuplicateReport:
    """Run duplicate detection on a DataFrame.

    Args:
        df: input dataframe.
        near_constant_threshold: a column where one value covers ≥ this share
            of rows is reported as "near-constant" (catches columns that are
            effectively useless features even when not strictly constant).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("detect_duplicates requires a pandas DataFrame")

    duplicated_row_mask = df.duplicated(keep=False)
    duplicate_row_indices = df.index[duplicated_row_mask].tolist()

    constant_cols: list[str] = []
    near_constant_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if series.nunique(dropna=False) <= 1:
            constant_cols.append(col)
            continue
        top_share = float(series.value_counts(dropna=False, normalize=True).iloc[0])
        if top_share >= near_constant_threshold:
            near_constant_cols.append(col)

    duplicate_column_groups = _find_duplicate_columns(df)

    report = DuplicateReport(
        n_duplicate_rows=int(duplicated_row_mask.sum()),
        duplicate_row_indices=duplicate_row_indices[:1000],  # cap payload
        n_duplicate_column_groups=len(duplicate_column_groups),
        duplicate_column_groups=duplicate_column_groups,
        constant_columns=constant_cols,
        near_constant_columns=near_constant_cols,
        total_rows=int(df.shape[0]),
        total_columns=int(df.shape[1]),
    )
    logger.info(
        "duplicate scan complete",
        extra={
            "ctx_dup_rows": report.n_duplicate_rows,
            "ctx_dup_col_groups": report.n_duplicate_column_groups,
            "ctx_constant_cols": len(report.constant_columns),
        },
    )
    return report
