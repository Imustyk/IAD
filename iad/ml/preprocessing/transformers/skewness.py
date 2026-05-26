"""Skewness correction via Yeo-Johnson power transformation.

Why Yeo-Johnson and not Box-Cox?
    Box-Cox is undefined for zero or negative values; Yeo-Johnson handles the
    full real line. For ML use this matters — most "income"-like features
    contain zeros and "delta"-like features contain negatives.

The transformer fits a per-column power transform on columns whose absolute
skewness exceeds ``skew_threshold``. Columns below the threshold are passed
through unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from iad.ml.preprocessing.exceptions import TransformerNotFittedError


class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """Apply Yeo-Johnson to highly skewed numeric columns.

    Args:
        columns: numeric columns to inspect. ``None`` → all numeric columns
            at fit time.
        skew_threshold: absolute skewness above which a column is power-
            transformed. The default of ``1.0`` is a common rule of thumb;
            tighter thresholds correct more aggressively.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        skew_threshold: float = 1.0,
    ) -> None:
        if skew_threshold < 0:
            raise ValueError("skew_threshold must be non-negative")
        self.columns = columns
        self.skew_threshold = skew_threshold
        self.skewed_columns_: list[str] = []
        self.transformer_: PowerTransformer | None = None

    def fit(self, X: pd.DataFrame, y=None) -> SkewnessCorrector:  # noqa: ARG002
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SkewnessCorrector requires a pandas DataFrame")

        candidates = (
            list(self.columns)
            if self.columns is not None
            else X.select_dtypes(include=["number"]).columns.tolist()
        )
        skewed: list[str] = []
        for col in candidates:
            if col not in X.columns:
                continue
            series = X[col].dropna().astype(float)
            if series.empty or series.nunique() < 2:
                continue
            if abs(series.skew()) >= self.skew_threshold:
                skewed.append(col)
        self.skewed_columns_ = skewed

        if skewed:
            self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.transformer_.fit(X[skewed].fillna(X[skewed].median(numeric_only=True)).astype(float))
        else:
            self.transformer_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.skewed_columns_ is None:
            raise TransformerNotFittedError("SkewnessCorrector.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SkewnessCorrector requires a pandas DataFrame")
        out = X.copy()
        if not self.skewed_columns_ or self.transformer_ is None:
            return out
        present = [c for c in self.skewed_columns_ if c in out.columns]
        if not present:
            return out
        # Fill NaNs with column median so PowerTransformer doesn't choke;
        # downstream imputers can still re-handle these if needed.
        sub = out[present].astype(float)
        sub = sub.fillna(sub.median(numeric_only=True))
        transformed = self.transformer_.transform(sub.values)
        out[present] = transformed
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            return np.array(self.skewed_columns_)
        return np.asarray(input_features)
