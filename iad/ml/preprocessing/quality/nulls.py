"""Null analysis with configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from iad.core.logging import get_logger

logger = get_logger("iad.ml.preprocessing.quality.nulls")


@dataclass(frozen=True)
class NullReport:
    """Per-column null statistics."""

    columns: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    shares: list[float] = field(default_factory=list)
    total_rows: int = 0
    threshold: float = 0.5

    @property
    def n_above_threshold(self) -> int:
        return sum(1 for s in self.shares if s >= self.threshold)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "column": self.columns,
                "n_missing": self.counts,
                "share_missing": self.shares,
                "above_threshold": [s >= self.threshold for s in self.shares],
            }
        )
        return df.sort_values("share_missing", ascending=False).reset_index(drop=True)

    def columns_above_threshold(self) -> list[str]:
        return [c for c, s in zip(self.columns, self.shares) if s >= self.threshold]


def null_report(df: pd.DataFrame, threshold: float = 0.5) -> NullReport:
    """Compute null share per column.

    Args:
        df: input dataframe.
        threshold: share above which a column is flagged. ``0.5`` flags any
            column where >= half the rows are missing.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("null_report requires a pandas DataFrame")
    n = max(df.shape[0], 1)
    columns = df.columns.tolist()
    counts = [int(df[c].isna().sum()) for c in columns]
    shares = [round(c / n, 6) for c in counts]

    report = NullReport(
        columns=columns,
        counts=counts,
        shares=shares,
        total_rows=int(df.shape[0]),
        threshold=float(threshold),
    )
    logger.info(
        "null analysis complete",
        extra={
            "ctx_threshold": threshold,
            "ctx_above_threshold": report.n_above_threshold,
        },
    )
    return report


def columns_above_null_threshold(df: pd.DataFrame, threshold: float) -> list[str]:
    """Convenience wrapper returning just the offending columns."""
    return null_report(df, threshold).columns_above_threshold()
