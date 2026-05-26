"""Phase 11 forecasting tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.forecasting.arima import forecast_arima
from iad.ml.forecasting.decomposition import decompose_series


@pytest.fixture
def ts_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    values = [10 + i * 0.5 + (i % 12) for i in range(len(dates))]
    return pd.DataFrame({"month": dates, "sales": values})


@pytest.mark.unit
def test_decompose(ts_df) -> None:
    report = decompose_series(
        ts_df, datetime_column="month", value_column="sales", period=12
    )
    assert len(report.observed) == len(ts_df)


@pytest.mark.unit
def test_arima_forecast(ts_df) -> None:
    report = forecast_arima(
        ts_df,
        datetime_column="month",
        value_column="sales",
        horizon=6,
        order=(1, 0, 0),
    )
    assert len(report.forecast) == 6


@pytest.mark.unit
def test_prophet_forecast_optional(ts_df) -> None:
    pytest.importorskip("prophet")
    from iad.ml.forecasting.prophet_model import forecast_prophet

    report = forecast_prophet(
        ts_df,
        datetime_column="month",
        value_column="sales",
        horizon=4,
    )
    assert len(report.forecast) == 4
