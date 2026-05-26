"""Dataset profiling — quality score + issues + simple HTML render.

The full ydata-profiling integration is deferred to Phase 13 (export/reporting).
This profiler computes the inputs that the dashboard cards will surface
on the home page in Phase 4.

The "quality score" is a heuristic 0-1 composite:

* 0.40 × (1 − missing share)
* 0.20 × (1 − duplicate share)
* 0.20 × (1 − constant-column share)
* 0.20 × (1 − high-skew share)

Higher = healthier. Known limitation: the score is not calibrated; it is
designed for relative comparison between two snapshots of the same dataset
or between sibling datasets, not as an absolute SLA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from iad.core.logging import get_logger
from iad.ml.preprocessing._dtypes import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
)
from iad.ml.preprocessing.quality.duplicates import detect_duplicates
from iad.ml.preprocessing.quality.nulls import null_report

logger = get_logger("iad.ml.preprocessing.profiling")


@dataclass(frozen=True)
class DataProfile:
    """Compact description of a dataset."""

    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    memory_bytes: int
    missing_cells: int
    missing_share: float
    duplicate_rows: int
    constant_columns: list[str]
    high_skew_columns: list[str]
    high_cardinality_columns: list[str]
    quality_score: float
    issues: list[str] = field(default_factory=list)
    column_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "n_datetime": self.n_datetime,
            "memory_bytes": self.memory_bytes,
            "missing_cells": self.missing_cells,
            "missing_share": self.missing_share,
            "duplicate_rows": self.duplicate_rows,
            "constant_columns": self.constant_columns,
            "high_skew_columns": self.high_skew_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "column_summary": self.column_summary,
        }


class DataProfiler:
    """Compute a :class:`DataProfile` and optionally render it as HTML."""

    def __init__(
        self,
        *,
        skewness_threshold: float = 1.0,
        cardinality_threshold: int = 50,
    ) -> None:
        if skewness_threshold < 0:
            raise ValueError("skewness_threshold must be non-negative")
        if cardinality_threshold < 1:
            raise ValueError("cardinality_threshold must be >= 1")
        self.skewness_threshold = skewness_threshold
        self.cardinality_threshold = cardinality_threshold

    # ------------------------------------------------------------------
    def profile(self, df: pd.DataFrame) -> DataProfile:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataProfiler.profile requires a pandas DataFrame")
        n_rows, n_columns = df.shape
        memory_bytes = int(df.memory_usage(deep=True).sum())

        nulls = null_report(df, threshold=0.5)
        dupes = detect_duplicates(df)

        numeric_cols = numeric_columns(df)
        datetime_cols = datetime_columns(df)
        categorical_cols = categorical_columns(df)

        high_skew_cols: list[str] = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 1 and abs(series.skew()) >= self.skewness_threshold:
                high_skew_cols.append(col)

        high_cardinality_cols: list[str] = []
        for col in categorical_cols:
            if df[col].nunique(dropna=True) > self.cardinality_threshold:
                high_cardinality_cols.append(col)

        missing_cells = int(df.isna().sum().sum())
        missing_share = missing_cells / max(n_rows * max(n_columns, 1), 1)
        dup_share = dupes.duplicate_row_share
        constant_share = len(dupes.constant_columns) / max(n_columns, 1)
        skew_share = len(high_skew_cols) / max(len(numeric_cols), 1)

        score = (
            0.40 * (1.0 - missing_share)
            + 0.20 * (1.0 - dup_share)
            + 0.20 * (1.0 - constant_share)
            + 0.20 * (1.0 - skew_share)
        )
        score = float(max(0.0, min(1.0, score)))

        issues: list[str] = []
        if missing_share > 0.1:
            issues.append(f"{missing_share:.1%} of cells are missing")
        if nulls.n_above_threshold:
            issues.append(
                f"{nulls.n_above_threshold} column(s) above the 50% null threshold"
            )
        if dup_share > 0:
            issues.append(f"{dupes.n_duplicate_rows} duplicate row(s)")
        if dupes.constant_columns:
            issues.append(f"constant columns: {dupes.constant_columns}")
        if dupes.duplicate_column_groups:
            issues.append(
                f"{len(dupes.duplicate_column_groups)} duplicate column group(s)"
            )
        if high_cardinality_cols:
            issues.append(
                f"high-cardinality categorical columns: {high_cardinality_cols[:5]}"
            )
        if high_skew_cols:
            issues.append(f"highly skewed numeric columns: {high_skew_cols[:5]}")

        column_summary = [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        ]

        profile = DataProfile(
            n_rows=int(n_rows),
            n_columns=int(n_columns),
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            n_datetime=len(datetime_cols),
            memory_bytes=memory_bytes,
            missing_cells=missing_cells,
            missing_share=round(missing_share, 6),
            duplicate_rows=dupes.n_duplicate_rows,
            constant_columns=dupes.constant_columns,
            high_skew_columns=high_skew_cols,
            high_cardinality_columns=high_cardinality_cols,
            quality_score=round(score, 4),
            issues=issues,
            column_summary=column_summary,
        )
        logger.info(
            "data profile computed",
            extra={
                "ctx_quality_score": profile.quality_score,
                "ctx_n_issues": len(profile.issues),
            },
        )
        return profile

    # ------------------------------------------------------------------
    def to_html(self, profile: DataProfile, *, title: str = "Data Profile") -> str:
        """Render a self-contained HTML report — no external assets."""
        rows = "".join(
            f"<tr><td>{escape(str(c['column']))}</td>"
            f"<td>{escape(str(c['dtype']))}</td>"
            f"<td>{c['missing']}</td>"
            f"<td>{c['unique']}</td></tr>"
            for c in profile.column_summary
        )
        issues = "".join(f"<li>{escape(i)}</li>" for i in profile.issues) or "<li>None</li>"
        score_pct = int(profile.quality_score * 100)
        return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>{escape(title)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2rem auto; color: #111; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 1rem; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
  .kpi {{ background: #f3f4f6; border-radius: 8px; padding: 1rem; }}
  .kpi h2 {{ font-size: 0.8rem; text-transform: uppercase; color: #6b7280; margin: 0; letter-spacing: 0.06em; }}
  .kpi p {{ font-size: 1.4rem; margin: 0.25rem 0 0; font-weight: 600; }}
  .score {{ background: linear-gradient(90deg, #22c55e {score_pct}%, #e5e7eb {score_pct}%); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #e5e7eb; font-size: 0.9rem; }}
  ul {{ margin: 0.5rem 0 1rem 1.25rem; }}
</style></head><body>
  <h1>{escape(title)}</h1>
  <div class='kpis'>
    <div class='kpi score'><h2>Quality score</h2><p>{score_pct}%</p></div>
    <div class='kpi'><h2>Rows</h2><p>{profile.n_rows:,}</p></div>
    <div class='kpi'><h2>Columns</h2><p>{profile.n_columns:,}</p></div>
    <div class='kpi'><h2>Missing cells</h2><p>{profile.missing_cells:,}</p></div>
  </div>
  <h2>Issues</h2><ul>{issues}</ul>
  <h2>Columns</h2>
  <table><thead><tr><th>Column</th><th>Dtype</th><th>Missing</th><th>Unique</th></tr></thead>
  <tbody>{rows}</tbody></table>
</body></html>"""

    def to_html_file(self, profile: DataProfile, path: Path | str, *, title: str = "Data Profile") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(profile, title=title), encoding="utf-8")
        return path
