"""Recommendation system results."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RecommendationReport:
    """Top-N recommendations for a user or item."""

    method: str
    target_user: str | int | None
    recommendations: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    similarity_matrix: pd.DataFrame | None = None
