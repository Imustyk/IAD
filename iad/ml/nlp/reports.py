"""Structured NLP analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SentimentReport:
    """Per-row sentiment scores and aggregate distribution."""

    column: str
    scores: pd.DataFrame
    summary: dict[str, float]
    method: str = "vader"

    def distribution_frame(self) -> pd.DataFrame:
        if "label" not in self.scores.columns:
            return pd.DataFrame()
        return (
            self.scores["label"]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
            .rename(columns={"index": "label"})
        )


@dataclass(frozen=True)
class EmbeddingReport:
    """Low-dimensional embedding matrix for visualization."""

    column: str
    matrix: pd.DataFrame
    method: str
    n_features: int
    model_name: str | None = None


@dataclass(frozen=True)
class TopicReport:
    """LDA topic model output."""

    column: str
    n_topics: int
    topics: list[dict[str, object]] = field(default_factory=list)
    document_topics: pd.DataFrame | None = None
    method: str = "lda"
