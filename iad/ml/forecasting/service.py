"""Forecasting service — decomposition, ARIMA, Prophet."""
from __future__ import annotations

import pandas as pd

from iad.ml.forecasting.arima import forecast_arima
from iad.ml.forecasting.decomposition import decompose_series
from iad.ml.forecasting.prophet_model import forecast_prophet
from iad.ml.forecasting.reports import DecompositionReport, ForecastReport


class ForecastingService:
    """Time-series analytics facade."""

    def decompose(
        self,
        df: pd.DataFrame,
        *,
        datetime_column: str,
        value_column: str,
        period: int | None = None,
    ) -> DecompositionReport:
        return decompose_series(
            df,
            datetime_column=datetime_column,
            value_column=value_column,
            period=period,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        datetime_column: str,
        value_column: str,
        method: str = "arima",
        horizon: int = 14,
        arima_order: tuple[int, int, int] = (1, 1, 1),
        holdout: int | None = None,
    ) -> ForecastReport:
        if method == "arima":
            return forecast_arima(
                df,
                datetime_column=datetime_column,
                value_column=value_column,
                horizon=horizon,
                order=arima_order,
                holdout=holdout,
            )
        if method == "prophet":
            return forecast_prophet(
                df,
                datetime_column=datetime_column,
                value_column=value_column,
                horizon=horizon,
                holdout=holdout,
            )
        from iad.core.exceptions import AnalyticsError

        raise AnalyticsError(
            f"Unknown forecast method {method!r}.",
            user_message="Choose arima or prophet.",
        )
