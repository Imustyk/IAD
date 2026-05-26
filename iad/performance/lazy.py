"""Lazy dataset views — paginated previews without copying full frames."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from iad.performance.memory import MemoryFootprint, sample_if_large


@dataclass
class LazyDatasetView:
    """Lightweight handle for UI previews of large in-memory datasets."""

    full: pd.DataFrame
    preview: pd.DataFrame
    was_sampled: bool
    footprint: MemoryFootprint
    total_rows: int

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, preview_rows: int | None = None) -> LazyDatasetView:
        preview, sampled = sample_if_large(df, max_rows=preview_rows)
        return cls(
            full=df,
            preview=preview,
            was_sampled=sampled,
            footprint=MemoryFootprint.from_dataframe(df),
            total_rows=len(df),
        )

    def page(self, page: int, page_size: int = 100) -> pd.DataFrame:
        start = page * page_size
        end = start + page_size
        return self.full.iloc[start:end]

    def summary_line(self) -> str:
        base = f"{self.total_rows:,} rows × {self.footprint.columns:,} cols · {self.footprint.memory_mb} MB"
        if self.was_sampled:
            return f"{base} · preview shows {len(self.preview):,} rows"
        return base
