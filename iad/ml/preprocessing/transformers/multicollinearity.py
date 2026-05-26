"""Drop redundant numeric columns by correlation magnitude.

For each pair of numeric features whose absolute Pearson correlation exceeds
``threshold``, the second is dropped (the first is kept by column order).
This is the cheap, deterministic alternative to VIF — fast, easy to explain
to a stakeholder, and good enough for the vast majority of tabular tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from iad.ml.preprocessing.exceptions import TransformerNotFittedError


class MulticollinearityReducer(BaseEstimator, TransformerMixin):
    """Remove highly correlated numeric features.

    Args:
        threshold: absolute correlation threshold. ``0.95`` keeps everything
            unless it is essentially a linear copy of another feature.
        method: one of ``"pearson"``, ``"spearman"``, ``"kendall"``.
        protect: never drop these columns (e.g. business-critical KPIs).
    """

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        method: str = "pearson",
        protect: list[str] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("method must be one of pearson/spearman/kendall")
        # sklearn contract: store __init__ parameters verbatim. Any
        # normalisation of None / mutable defaults must happen in fit().
        self.threshold = threshold
        self.method = method
        self.protect = protect
        self.dropped_: list[str] = []
        self.kept_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> MulticollinearityReducer:  # noqa: ARG002
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MulticollinearityReducer requires a pandas DataFrame")
        protected_list: list[str] = list(self.protect or [])
        numeric = X.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            self.dropped_ = []
            self.kept_ = X.columns.tolist()
            return self

        corr = numeric.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        protected = set(protected_list)

        to_drop: list[str] = []
        for col in upper.columns:
            if col in protected or col in to_drop:
                continue
            collinear = upper.index[upper[col].fillna(0) > self.threshold].tolist()
            for partner in collinear:
                if partner in protected or partner in to_drop:
                    continue
                # Drop the later of the two (col-order tie break).
                later = partner if list(upper.columns).index(partner) > list(upper.columns).index(col) else col
                to_drop.append(later)

        self.dropped_ = sorted(set(to_drop))
        self.kept_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kept_ is None:
            raise TransformerNotFittedError("MulticollinearityReducer.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MulticollinearityReducer requires a pandas DataFrame")
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            return np.array(self.kept_)
        kept = [c for c in input_features if c not in self.dropped_]
        return np.array(kept)
