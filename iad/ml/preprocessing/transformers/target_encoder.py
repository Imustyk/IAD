"""Smoothed K-fold target encoding for high-cardinality categoricals.

A correct target encoder has to address two failure modes:

1. **Leakage.** During training the encoded value of a row must not be
   computed using its own target. We solve this with K-fold out-of-fold
   encoding: row ``i`` is encoded using the mean computed over the other
   ``K-1`` folds.

2. **Cold start.** Categories with very few rows have a noisy mean. We
   blend each category mean with the global mean using a Bayesian shrinkage
   weight controlled by ``smoothing``.

The transformer follows sklearn semantics:

* ``fit(X, y)``           — learn the *full-data* mapping (used at inference).
* ``fit_transform(X, y)`` — return out-of-fold encoded training data.
* ``transform(X)``        — encode *new* data using the full-data mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from iad.ml.preprocessing._dtypes import is_categorical_like
from iad.ml.preprocessing.exceptions import PreprocessingError, TransformerNotFittedError


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """K-fold out-of-fold smoothed target encoder.

    Args:
        columns: columns to encode. ``None`` → every object/categorical
            column at fit time.
        smoothing: shrinkage strength. Higher ⇒ more weight on the global
            mean for sparse categories.
        n_folds: number of folds for OOF training-time encoding.
        random_state: reproducibility seed for the K-fold split.
        unseen_value: encoded value for categories never seen at fit time.
            ``None`` (default) → the global target mean.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        smoothing: float = 10.0,
        n_folds: int = 5,
        random_state: int = 42,
        unseen_value: float | None = None,
        suffix: str = "__te",
    ) -> None:
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.columns = columns
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state
        self.unseen_value = unseen_value
        self.suffix = suffix
        self._mappings_: dict[str, dict] = {}
        self._global_mean_: float | None = None
        self._fitted_columns_: list[str] = []

    # ------------------------------------------------------------------
    # Mapping computation
    # ------------------------------------------------------------------
    def _smoothed_mapping(self, x: pd.Series, y: pd.Series) -> tuple[dict, float]:
        global_mean = float(y.mean())
        df = pd.DataFrame({"x": x, "y": y}).dropna(subset=["x"])
        if df.empty:
            return {}, global_mean
        agg = df.groupby("x", dropna=False)["y"].agg(["count", "mean"])
        smoothed = (
            agg["count"] * agg["mean"] + self.smoothing * global_mean
        ) / (agg["count"] + self.smoothing)
        return smoothed.to_dict(), global_mean

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> SmoothedTargetEncoder:
        if y is None:
            raise PreprocessingError("SmoothedTargetEncoder.fit requires y")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmoothedTargetEncoder requires a pandas DataFrame")

        y_series = self._coerce_y(y, X.index)
        cols = self._select_columns(X)

        mappings: dict[str, dict] = {}
        global_means: list[float] = []
        for col in cols:
            mapping, gmean = self._smoothed_mapping(X[col], y_series)
            mappings[col] = mapping
            global_means.append(gmean)

        self._mappings_ = mappings
        self._global_mean_ = float(np.mean(global_means)) if global_means else float(y_series.mean())
        self._fitted_columns_ = cols
        return self

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:  # noqa: ARG002
        if y is None:
            raise PreprocessingError("SmoothedTargetEncoder.fit_transform requires y")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmoothedTargetEncoder requires a pandas DataFrame")
        y_series = self._coerce_y(y, X.index)

        # Fit the *final* mapping on the full data so transform() works later.
        self.fit(X, y_series)
        cols = self._fitted_columns_
        if not cols:
            return X.copy()

        out = X.copy()
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for col in cols:
            encoded = np.full(len(X), fill_value=self._global_mean_, dtype=float)
            for train_idx, test_idx in kf.split(X):
                # OOF mapping computed on the training split only.
                mapping, gmean = self._smoothed_mapping(
                    X[col].iloc[train_idx], y_series.iloc[train_idx]
                )
                fallback = self.unseen_value if self.unseen_value is not None else gmean
                fold_values = X[col].iloc[test_idx].map(mapping).fillna(fallback).to_numpy()
                encoded[test_idx] = fold_values
            out[col + self.suffix] = encoded
        return out

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns_:
            raise TransformerNotFittedError("SmoothedTargetEncoder.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmoothedTargetEncoder requires a pandas DataFrame")
        out = X.copy()
        fallback = self.unseen_value if self.unseen_value is not None else self._global_mean_
        for col in self._fitted_columns_:
            mapping = self._mappings_.get(col, {})
            if col not in out.columns:
                out[col + self.suffix] = fallback
                continue
            out[col + self.suffix] = out[col].map(mapping).fillna(fallback)
        return out

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [c + self.suffix for c in self._fitted_columns_]
        if input_features is not None:
            base = list(input_features)
            return np.array(base + names)
        return np.array(names)

    # ------------------------------------------------------------------
    def _select_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return [c for c in X.columns if is_categorical_like(X[c])]
        return [c for c in self.columns if c in X.columns]

    @staticmethod
    def _coerce_y(y, index) -> pd.Series:
        if isinstance(y, pd.Series):
            return y.astype(float)
        arr = np.asarray(y, dtype=float)
        return pd.Series(arr, index=index, name="target")
