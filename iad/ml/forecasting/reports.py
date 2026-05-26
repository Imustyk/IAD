"""Time-series forecasting results."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DecompositionReport:
    """Seasonal decomposition components."""

    datetime_column: str
    value_column: str
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int


@dataclass(frozen=True)
class ForecastReport:
    """Point forecasts with optional confidence intervals."""

    datetime_column: str
    value_column: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    method: str
    metrics: dict[str, float] = field(default_factory=dict)
    model_summary: str | None = None
