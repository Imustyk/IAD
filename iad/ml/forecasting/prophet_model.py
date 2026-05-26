"""Facebook Prophet forecasting (optional dependency)."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error

from iad.core.exceptions import AnalyticsError
from iad.core.logging import get_logger
from iad.ml.forecasting.availability import prophet_available
from iad.ml.forecasting.prepare import prepare_series
from iad.ml.forecasting.reports import ForecastReport

logger = get_logger("iad.ml.forecasting.prophet")


def forecast_prophet(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    value_column: str,
    horizon: int = 14,
    holdout: int | None = None,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> ForecastReport:
    """Fit Prophet and return forecast with uncertainty intervals."""
    if not prophet_available():
        raise AnalyticsError(
            "prophet is not installed.",
            user_message="Install forecasting extras: pip install prophet",
        )

    from prophet import Prophet

    series = prepare_series(df, datetime_column=datetime_column, value_column=value_column)
    if len(series) < 10:
        raise AnalyticsError(
            "Need at least 10 observations for Prophet.",
            user_message="Load more historical data.",
        )

    train = series
    test = None
    if holdout and holdout > 0 and holdout < len(series) // 2:
        train = series.iloc[:-holdout]
        test = series.iloc[-holdout:]

    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon, freq=pd.infer_freq(train.index))
    forecast = model.predict(future)
    forecast_tail = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")

    in_sample = model.predict(train_df.rename(columns={"ds": "ds"}))[["ds", "yhat"]]
    history_df = pd.DataFrame({"actual": series})
    history_df["fitted"] = in_sample.set_index("ds")["yhat"].reindex(series.index)
    history_df.index.name = datetime_column

    metrics: dict[str, float] = {}
    if test is not None and len(test) > 0:
        holdout_df = test.reset_index()
        holdout_df.columns = ["ds", "y"]
        pred = model.predict(holdout_df)
        metrics["holdout_mae"] = float(mean_absolute_error(test.values, pred["yhat"]))
        metrics["holdout_rmse"] = float(
            ((test.values - pred["yhat"].values) ** 2).mean() ** 0.5
        )

    logger.info("prophet forecast", extra={"horizon": horizon})
    return ForecastReport(
        datetime_column=datetime_column,
        value_column=value_column,
        history=history_df,
        forecast=forecast_tail,
        method="prophet",
        metrics=metrics,
    )
