"""Extract calendar features from datetime columns.

The transformer is autodetecting: when ``columns=None`` it picks every
``datetime64[*]`` column at fit time. Each datetime column ``c`` is replaced
(or supplemented) by the following numeric columns:

* ``c__year``, ``c__month``, ``c__day``,
* ``c__hour``, ``c__minute``,
* ``c__dayofweek``, ``c__weekofyear``, ``c__quarter``,
* ``c__is_weekend`` (0/1),
* ``c__day_of_year``.

These are the features almost every business problem needs from a timestamp
(seasonality, weekday/weekend behaviour, intraday patterns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from iad.ml.preprocessing.exceptions import TransformerNotFittedError

_DATETIME_PARTS = (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "dayofweek",
    "weekofyear",
    "quarter",
    "is_weekend",
    "day_of_year",
)


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible datetime feature extractor.

    Args:
        columns: explicit datetime columns. ``None`` → auto-detect at fit.
        drop_original: if True (default) the original datetime columns are
            removed from the output frame.
        coerce: when True, non-datetime columns named in ``columns`` are
            coerced via :func:`pandas.to_datetime` (with ``errors='coerce'``).
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        drop_original: bool = True,
        coerce: bool = True,
    ) -> None:
        self.columns = columns
        self.drop_original = drop_original
        self.coerce = coerce
        self._fitted_columns_: list[str] | None = None
        self._feature_names_out_: list[str] | None = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> DatetimeFeatureExtractor:  # noqa: ARG002
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DatetimeFeatureExtractor requires a pandas DataFrame")
        if self.columns is None:
            datetime_cols = [
                c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])
            ]
        else:
            datetime_cols = list(self.columns)
        self._fitted_columns_ = datetime_cols
        self._feature_names_out_ = self._build_output_names(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fitted_columns_ is None:
            raise TransformerNotFittedError("DatetimeFeatureExtractor.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DatetimeFeatureExtractor requires a pandas DataFrame")

        out = X.copy()
        for col in self._fitted_columns_:
            if col not in out.columns:
                # Tolerate missing column at inference time — fill features with NaN.
                for part in _DATETIME_PARTS:
                    out[f"{col}__{part}"] = np.nan
                continue
            series = out[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                if self.coerce:
                    series = pd.to_datetime(series, errors="coerce")
                else:
                    raise TypeError(
                        f"column {col!r} is not datetime and coerce=False"
                    )

            out[f"{col}__year"] = series.dt.year.astype("Int64")
            out[f"{col}__month"] = series.dt.month.astype("Int64")
            out[f"{col}__day"] = series.dt.day.astype("Int64")
            out[f"{col}__hour"] = series.dt.hour.astype("Int64")
            out[f"{col}__minute"] = series.dt.minute.astype("Int64")
            out[f"{col}__dayofweek"] = series.dt.dayofweek.astype("Int64")
            out[f"{col}__weekofyear"] = (
                series.dt.isocalendar().week.astype("Int64")
                if hasattr(series.dt, "isocalendar")
                else series.dt.weekofyear.astype("Int64")  # pragma: no cover
            )
            out[f"{col}__quarter"] = series.dt.quarter.astype("Int64")
            out[f"{col}__is_weekend"] = (series.dt.dayofweek >= 5).astype("Int64")
            out[f"{col}__day_of_year"] = series.dt.dayofyear.astype("Int64")

            if self.drop_original:
                out = out.drop(columns=[col])
        return out

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        if self._feature_names_out_ is None:
            raise TransformerNotFittedError("DatetimeFeatureExtractor.fit() not called")
        return np.array(self._feature_names_out_)

    def _build_output_names(self, X: pd.DataFrame) -> list[str]:
        cols = self._fitted_columns_ or []
        kept = [c for c in X.columns if c not in cols] if self.drop_original else list(X.columns)
        derived = [f"{c}__{part}" for c in cols for part in _DATETIME_PARTS]
        return kept + derived
