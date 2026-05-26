"""ARIMA forecasting via statsmodels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from iad.core.exceptions import AnalyticsError
from iad.core.logging import get_logger
from iad.ml.forecasting.prepare import prepare_series
from iad.ml.forecasting.reports import ForecastReport

logger = get_logger("iad.ml.forecasting.arima")


def forecast_arima(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    value_column: str,
    horizon: int = 14,
    order: tuple[int, int, int] = (1, 1, 1),
    holdout: int | None = None,
) -> ForecastReport:
    """Fit ARIMA and produce in-sample + future forecasts."""
    series = prepare_series(df, datetime_column=datetime_column, value_column=value_column)
    if len(series) < max(10, sum(order) + 5):
        raise AnalyticsError(
            "Time series too short for ARIMA.",
            user_message="Add more historical points or simplify the ARIMA order.",
        )

    train = series
    test = None
    if holdout and holdout > 0 and holdout < len(series) // 2:
        train = series.iloc[:-holdout]
        test = series.iloc[-holdout:]

    try:
        fitted = ARIMA(train, order=order).fit()
    except Exception as exc:
        raise AnalyticsError(
            f"ARIMA fit failed: {exc}",
            user_message="Try a simpler order such as (1,1,1) or check for constant series.",
        ) from exc

    in_sample = fitted.fittedvalues
    future = fitted.forecast(steps=horizon)
    conf = fitted.get_forecast(steps=horizon).conf_int(alpha=0.05)

    last_ts = train.index[-1]
    freq = pd.infer_freq(train.index)
    if freq:
        future_index = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
    else:
        step = (train.index[-1] - train.index[-2]) if len(train) > 1 else pd.Timedelta(days=1)
        future_index = [last_ts + step * (i + 1) for i in range(horizon)]

    forecast_df = pd.DataFrame(
        {
            "yhat": future.values if hasattr(future, "values") else future,
            "yhat_lower": conf.iloc[:, 0].values,
            "yhat_upper": conf.iloc[:, 1].values,
        },
        index=future_index,
    )
    forecast_df.index.name = datetime_column

    history_df = pd.DataFrame(
        {
            "actual": series,
            "fitted": in_sample.reindex(series.index),
        }
    )
    history_df.index.name = datetime_column

    metrics: dict[str, float] = {}
    if test is not None and len(test) > 0:
        test_pred = fitted.forecast(steps=len(test))
        metrics["holdout_mae"] = float(mean_absolute_error(test, test_pred))
        metrics["holdout_rmse"] = float(np.sqrt(mean_squared_error(test, test_pred)))

    logger.info("arima forecast", extra={"horizon": horizon, "order": order})
    return ForecastReport(
        datetime_column=datetime_column,
        value_column=value_column,
        history=history_df,
        forecast=forecast_df,
        method=f"arima{order}",
        metrics=metrics,
        model_summary=str(fitted.summary()),
    )
