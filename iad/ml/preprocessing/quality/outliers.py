"""Univariate and multivariate outlier detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from iad.core.logging import get_logger

logger = get_logger("iad.ml.preprocessing.quality.outliers")


@dataclass(frozen=True)
class OutlierReport:
    """Per-method, per-column outlier summary."""

    method: Literal["iqr", "zscore", "isolation_forest"]
    columns: list[str] = field(default_factory=list)
    n_outliers: list[int] = field(default_factory=list)
    shares: list[float] = field(default_factory=list)
    bounds: list[tuple[float, float] | None] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": self.columns,
                "n_outliers": self.n_outliers,
                "share": self.shares,
                "lower_bound": [b[0] if b else None for b in self.bounds],
                "upper_bound": [b[1] if b else None for b in self.bounds],
                "method": self.method,
            }
        ).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def detect_outliers_iqr(
    df: pd.DataFrame, columns: list[str] | None = None, k: float = 1.5
) -> OutlierReport:
    """Tukey IQR outlier detection — column-wise.

    Args:
        df: input dataframe.
        columns: numeric columns to check. ``None`` → all numeric columns.
        k: multiplier for the IQR. ``1.5`` is the classic Tukey rule.
    """
    numeric = df.select_dtypes(include=["number"])
    cols = list(columns) if columns is not None else numeric.columns.tolist()

    n_outliers: list[int] = []
    shares: list[float] = []
    bounds: list[tuple[float, float] | None] = []
    for col in cols:
        if col not in numeric.columns:
            n_outliers.append(0)
            shares.append(0.0)
            bounds.append(None)
            continue
        series = numeric[col].dropna()
        if series.empty:
            n_outliers.append(0)
            shares.append(0.0)
            bounds.append(None)
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = float(q1 - k * iqr), float(q3 + k * iqr)
        mask = (series < lo) | (series > hi)
        n_outliers.append(int(mask.sum()))
        shares.append(round(float(mask.mean()), 6))
        bounds.append((lo, hi))

    return OutlierReport("iqr", cols, n_outliers, shares, bounds)


def detect_outliers_zscore(
    df: pd.DataFrame, columns: list[str] | None = None, threshold: float = 3.0
) -> OutlierReport:
    """Z-score outlier detection — column-wise."""
    numeric = df.select_dtypes(include=["number"])
    cols = list(columns) if columns is not None else numeric.columns.tolist()

    n_outliers: list[int] = []
    shares: list[float] = []
    bounds: list[tuple[float, float] | None] = []
    for col in cols:
        if col not in numeric.columns:
            n_outliers.append(0)
            shares.append(0.0)
            bounds.append(None)
            continue
        series = numeric[col].dropna()
        if len(series) < 2 or series.std(ddof=0) == 0:
            n_outliers.append(0)
            shares.append(0.0)
            bounds.append(None)
            continue
        z = (series - series.mean()) / series.std(ddof=0)
        mask = z.abs() > threshold
        mu, sigma = float(series.mean()), float(series.std(ddof=0))
        n_outliers.append(int(mask.sum()))
        shares.append(round(float(mask.mean()), 6))
        bounds.append((mu - threshold * sigma, mu + threshold * sigma))

    return OutlierReport("zscore", cols, n_outliers, shares, bounds)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> OutlierReport:
    """Multivariate outlier detection via scikit-learn's IsolationForest."""
    from sklearn.ensemble import IsolationForest  # local import to keep cold-start fast

    numeric = df.select_dtypes(include=["number"])
    cols = list(columns) if columns is not None else numeric.columns.tolist()
    if not cols:
        return OutlierReport("isolation_forest", [], [], [], [])

    sub = numeric[cols].dropna()
    if sub.empty:
        return OutlierReport("isolation_forest", cols, [0] * len(cols), [0.0] * len(cols), [None] * len(cols))

    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = model.fit_predict(sub.values)
    is_outlier = preds == -1
    n_total = int(is_outlier.sum())

    # IsolationForest is multivariate so per-column counts are not the right
    # abstraction; we report the same total against each input column for
    # convenience and let the UI render an overall figure.
    counts = [n_total] * len(cols)
    shares = [round(n_total / max(len(sub), 1), 6)] * len(cols)
    bounds: list[tuple[float, float] | None] = [None] * len(cols)
    logger.info(
        "isolation forest outliers",
        extra={"ctx_n_total": n_total, "ctx_contamination": contamination, "ctx_cols": cols},
    )
    return OutlierReport("isolation_forest", cols, counts, shares, bounds)


def cap_outliers(
    df: pd.DataFrame, columns: list[str] | None = None, k: float = 1.5
) -> pd.DataFrame:
    """Winsorise columns at their IQR bounds. Returns a copy."""
    out = df.copy()
    numeric = out.select_dtypes(include=["number"])
    cols = list(columns) if columns is not None else numeric.columns.tolist()
    for col in cols:
        if col not in numeric.columns:
            continue
        series = out[col]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out[col] = np.clip(series, lo, hi)
    return out
