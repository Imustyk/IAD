"""Composite feature selector — variance + correlation + model-based importance.

Three filters run in order, each cheaper than the next:

1. **Variance filter** — drop columns whose variance is below a tolerance.
2. **Correlation filter** — drop one of each pair with |corr| > threshold.
3. **Model-based filter** — fit a tree ensemble and keep the top-N features
   by importance. ``max_features`` caps the final selection.

The selector is sklearn-compatible and supports ``get_feature_names_out``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from iad.ml.preprocessing.exceptions import PreprocessingError, TransformerNotFittedError


class AutoFeatureSelector(BaseEstimator, TransformerMixin):
    """Pipeline-friendly feature selector.

    Args:
        task: ``"classification"`` or ``"regression"`` for the model-based step.
        variance_threshold: drop columns with variance ≤ this. ``0.0`` keeps
            everything that is not strictly constant.
        correlation_threshold: |corr| above which one of a pair is dropped.
        max_features: hard cap on retained features; ``None`` disables.
        use_model_based: if True, fit a RandomForest to rank features. The
            top ``max_features`` (or 80 % of remaining features when
            ``max_features=None``) are kept.
        random_state: reproducibility seed for the embedded RandomForest.
    """

    def __init__(
        self,
        *,
        task: Literal["classification", "regression"] = "classification",
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        max_features: int | None = None,
        use_model_based: bool = True,
        random_state: int = 42,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if not 0 <= correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in [0, 1]")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be >= 1")
        self.task = task
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.use_model_based = use_model_based
        self.random_state = random_state
        self.selected_: list[str] | None = None
        self.report_: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> AutoFeatureSelector:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AutoFeatureSelector requires a pandas DataFrame")
        report: dict[str, list[str]] = {"low_variance": [], "correlated": [], "low_importance": []}
        candidates = X.columns.tolist()
        numeric_X = X.select_dtypes(include=["number"])

        # --- 1) Variance filter (numeric only) -------------------------
        if not numeric_X.empty:
            variances = numeric_X.var(axis=0).fillna(0.0)
            low_var = variances.index[variances <= self.variance_threshold].tolist()
            candidates = [c for c in candidates if c not in low_var]
            report["low_variance"] = low_var

        # --- 2) Correlation filter -------------------------------------
        numeric_candidates = [c for c in candidates if c in numeric_X.columns]
        if len(numeric_candidates) > 1:
            corr = numeric_X[numeric_candidates].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            corr_drop: list[str] = []
            for col in upper.columns:
                if col in corr_drop:
                    continue
                strong = upper.index[upper[col].fillna(0) > self.correlation_threshold].tolist()
                for partner in strong:
                    if partner in corr_drop:
                        continue
                    later = (
                        partner
                        if list(upper.columns).index(partner) > list(upper.columns).index(col)
                        else col
                    )
                    corr_drop.append(later)
            candidates = [c for c in candidates if c not in corr_drop]
            report["correlated"] = sorted(set(corr_drop))

        # --- 3) Model-based importance ---------------------------------
        if self.use_model_based and y is not None and candidates:
            try:
                ranking = self._rank_with_model(X[candidates], y)
                if ranking is not None:
                    if self.max_features is not None:
                        keep = ranking.head(self.max_features).index.tolist()
                    else:
                        # keep top 80% by importance (drop 20% trailing)
                        n_keep = max(1, int(np.ceil(0.8 * len(ranking))))
                        keep = ranking.head(n_keep).index.tolist()
                    dropped = [c for c in candidates if c not in keep]
                    report["low_importance"] = dropped
                    candidates = keep
            except Exception as exc:
                raise PreprocessingError(
                    f"AutoFeatureSelector model-based ranking failed: {exc}"
                ) from exc

        if self.max_features is not None and len(candidates) > self.max_features:
            candidates = candidates[: self.max_features]

        self.selected_ = candidates
        self.report_ = report
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_ is None:
            raise TransformerNotFittedError("AutoFeatureSelector.fit() not called")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AutoFeatureSelector requires a pandas DataFrame")
        return X[[c for c in self.selected_ if c in X.columns]].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        if self.selected_ is None:
            raise TransformerNotFittedError("AutoFeatureSelector.fit() not called")
        return np.array(self.selected_)

    # ------------------------------------------------------------------
    def _rank_with_model(self, X: pd.DataFrame, y) -> pd.Series | None:
        """Fit a RandomForest on the candidate features and return their importances."""
        # Encode categoricals via one-hot for the importance estimate only.
        encoded = pd.get_dummies(X, drop_first=False, dummy_na=False)
        encoded = encoded.fillna(encoded.median(numeric_only=True))
        if encoded.empty or encoded.shape[1] == 0:
            return None
        if self.task == "classification":
            model = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200, random_state=self.random_state, n_jobs=-1
            )
        model.fit(encoded.values, y)
        importances = pd.Series(model.feature_importances_, index=encoded.columns)
        # Aggregate one-hot importance back to source column.
        original_importance: dict[str, float] = {}
        for col in X.columns:
            mask = importances.index.str.startswith(f"{col}_") | (importances.index == col)
            original_importance[col] = float(importances[mask].sum())
        ranking = pd.Series(original_importance).sort_values(ascending=False)
        return ranking
