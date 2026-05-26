"""Trend / seasonality decomposition."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from iad.core.exceptions import AnalyticsError
from iad.core.logging import get_logger
from iad.ml.forecasting.prepare import prepare_series
from iad.ml.forecasting.reports import DecompositionReport

logger = get_logger("iad.ml.forecasting.decomposition")


def decompose_series(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    value_column: str,
    period: int | None = None,
    model: str = "additive",
) -> DecompositionReport:
    """Classical seasonal decomposition (moving averages)."""
    series = prepare_series(df, datetime_column=datetime_column, value_column=value_column)
    if len(series) < 4:
        raise AnalyticsError(
            "Need at least 4 observations for decomposition.",
            user_message="Load a longer time series.",
        )

    inferred_period = period
    if inferred_period is None:
        inferred_period = min(12, max(2, len(series) // 2))
    if len(series) < 2 * inferred_period:
        raise AnalyticsError(
            f"Series too short for period={inferred_period}.",
            user_message="Reduce seasonal period or add more history.",
        )

    result = seasonal_decompose(
        series,
        model=model,  # type: ignore[arg-type]
        period=inferred_period,
        extrapolate_trend="freq",
    )
    logger.info(
        "seasonal decomposition complete",
        extra={"period": inferred_period, "n": len(series)},
    )
    return DecompositionReport(
        datetime_column=datetime_column,
        value_column=value_column,
        observed=result.observed,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        period=inferred_period,
    )
