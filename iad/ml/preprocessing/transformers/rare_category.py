"""Group rare categorical values under a single placeholder label.

Rare categories blow up one-hot encoding and harm tree-based splits.
The grouper records the *kept* categories at fit time and at transform time
replaces anything outside that allow-list with ``replacement`` (default
``"Other"``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from iad.ml.preprocessing._dtypes import is_categorical_like
from iad.ml.preprocessing.exceptions import TransformerNotFittedError


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace categories below a frequency threshold with a placeholder.

    Args:
        columns: columns to process. ``None`` → every object/categorical
            column at fit time.
        min_frequency: minimum share of rows a category must cover to be
            kept. Categories below the cutoff become ``replacement``.
        max_categories: hard upper bound on kept categories per column. The
            top ``max_categories`` by frequency are kept, the rest grouped.
        replacement: label used for rare categories.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        min_frequency: float = 0.01,
        max_categories: int | None = None,
        replacement: str = "Other",
    ) -> None:
        if not 0 <= min_frequency <= 1:
            raise ValueError("min_frequency must be in [0, 1]")
        if max_categories is not None and max_categories < 1:
            raise ValueError("max_categories must be >= 1")
        self.columns = columns
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.replacement = replacement
        self.allowed_: dict[str, set] = {}
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> RareCategoryGrouper:  # noqa: ARG002
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RareCategoryGrouper requires a pandas DataFrame")
        if self.columns is None:
            cols = [c for c in X.columns if is_categorical_like(X[c])]
        else:
            cols = list(self.columns)

        allowed: dict[str, set] = {}
        for col in cols:
            if col not in X.columns:
                continue
            counts = X[col].value_counts(normalize=True, dropna=True)
            keep = counts[counts >= self.min_frequency].index
            if self.max_categories is not None:
                keep = keep[: self.max_categories]
            allowed[col] = set(keep)
        self.allowed_ = allowed
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise TransformerNotFittedError("RareCategoryGrouper.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RareCategoryGrouper requires a pandas DataFrame")
        out = X.copy()
        for col, allowed in self.allowed_.items():
            if col not in out.columns:
                continue
            mask = out[col].isin(allowed) | out[col].isna()
            out[col] = out[col].where(mask, other=self.replacement)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            return np.array(list(self.allowed_.keys()))
        return np.asarray(input_features)
