"""Distribution-distance metrics used by the drift detector.

All functions are pure: take two arrays / series, return a scalar (or scalar
+ p-value tuple). They never raise on empty input — they return ``np.nan``
so the detector can reason about ``flags`` instead of trapping exceptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon as _jensenshannon


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov (continuous)
# ---------------------------------------------------------------------------
def ks_statistic(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns:
        ``(statistic, p_value)``. The statistic is bounded in ``[0, 1]``;
        higher means more drift. ``p_value < 0.05`` traditionally means we
        reject the null of "same distribution".
    """
    ref = pd.Series(reference).dropna().to_numpy()
    cur = pd.Series(current).dropna().to_numpy()
    if len(ref) < 2 or len(cur) < 2:
        return float("nan"), float("nan")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Population Stability Index (continuous OR categorical)
# ---------------------------------------------------------------------------
def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
    categorical: bool = False,
) -> float:
    """Population Stability Index between two distributions.

    PSI is the de-facto standard in the credit-risk and ML-monitoring world.
    Interpretation rule of thumb (Siddiqi 2006):

    * ``< 0.1``       — no significant change
    * ``0.1 - 0.25``  — moderate shift
    * ``> 0.25``      — significant shift
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return float("nan")

    if categorical:
        categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        ref_dist = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
        cur_dist = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    else:
        ref_arr = ref.astype(float).to_numpy()
        cur_arr = cur.astype(float).to_numpy()
        # Build bin edges from the reference distribution so PSI is invariant
        # to the current sample size.
        edges = np.unique(np.quantile(ref_arr, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:  # all reference values identical
            return 0.0 if np.array_equal(np.unique(cur_arr), np.unique(ref_arr)) else float("inf")
        # Extend to (-inf, +inf) so out-of-range current values still bin.
        edges[0], edges[-1] = -np.inf, np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=edges)
        cur_counts, _ = np.histogram(cur_arr, bins=edges)
        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.clip(ref_dist, epsilon, None)
    cur_dist = np.clip(cur_dist, epsilon, None)
    psi = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
    return psi


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence (categorical or binned continuous)
# ---------------------------------------------------------------------------
def jensen_shannon_divergence(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    categorical: bool = True,
    bins: int = 10,
    base: float = 2.0,
) -> float:
    """Symmetric, bounded distance between two distributions.

    Returns a value in ``[0, 1]`` when ``base=2`` (the most common choice).
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return float("nan")

    if categorical:
        categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        p = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
        q = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        p_counts, _ = np.histogram(ref.astype(float), bins=edges)
        q_counts, _ = np.histogram(cur.astype(float), bins=edges)
        p = p_counts / max(p_counts.sum(), 1)
        q = q_counts / max(q_counts.sum(), 1)

    distance = float(_jensenshannon(p, q, base=base))
    return distance * distance  # JS divergence == squared JS distance


# ---------------------------------------------------------------------------
# Chi-square test for categorical columns
# ---------------------------------------------------------------------------
def chi_square_drift(
    reference: pd.Series, current: pd.Series
) -> tuple[float, float]:
    """Chi-square goodness-of-fit between current and reference distributions.

    Returns ``(statistic, p_value)``. Useful for low-cardinality categoricals
    where PSI's bin smoothing is overkill.
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return float("nan"), float("nan")
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    obs = cur.value_counts().reindex(categories, fill_value=0).to_numpy().astype(float)
    exp_share = ref.value_counts(normalize=True).reindex(categories, fill_value=1e-9).to_numpy()
    expected = exp_share * obs.sum()
    expected = np.clip(expected, 1e-9, None)
    statistic = float(((obs - expected) ** 2 / expected).sum())
    dof = max(len(categories) - 1, 1)
    p_value = float(1 - stats.chi2.cdf(statistic, df=dof))
    return statistic, p_value
