"""Diagnostic analytics: correlations, group comparisons, outliers, hypothesis tests."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


def correlation_matrix(df: pd.DataFrame, method: Literal["pearson", "spearman", "kendall"] = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method).round(4)


def top_correlations(df: pd.DataFrame, target: str | None = None, top_n: int = 15,
                     method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    corr = numeric.corr(method=method)

    if target and target in corr.columns:
        s = corr[target].drop(labels=[target]).abs().sort_values(ascending=False).head(top_n)
        return pd.DataFrame(
            {
                "feature": s.index,
                "correlation": [corr.loc[idx, target] for idx in s.index],
                "abs_correlation": s.values,
            }
        ).round(4)

    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pair_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])
    pair_df["abs_correlation"] = pair_df["correlation"].abs()
    return pair_df.sort_values("abs_correlation", ascending=False).head(top_n).round(4)


def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> dict:
    series = df[column].dropna()
    if series.empty:
        return {"column": column, "lower": np.nan, "upper": np.nan, "n_outliers": 0, "share_%": 0.0}
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return {
        "column": column,
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "n_outliers": int(mask.sum()),
        "share_%": round(float(mask.mean()) * 100, 2),
    }


def outliers_overview(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    rows = [detect_outliers_iqr(df, col, k=k) for col in df.select_dtypes(include=["number"]).columns]
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def group_comparison(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Group-level statistics for a numeric column."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(group_col, dropna=False)[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    return out.round(4)


def t_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Independent two-sample Welch t-test for binary groups."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"error": f"t-test requires exactly 2 groups, found {len(groups)}"}
    a = df.loc[df[group_col] == groups[0], value_col].dropna()
    b = df.loc[df[group_col] == groups[1], value_col].dropna()
    if len(a) < 2 or len(b) < 2:
        return {"error": "not enough observations in each group"}
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "test": "Welch t-test",
        "group_a": str(groups[0]),
        "group_b": str(groups[1]),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant_at_0.05": bool(p < 0.05),
    }


def anova_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """One-way ANOVA across multiple groups."""
    groups = [g.dropna().values for _, g in df.groupby(group_col, dropna=True)[value_col]]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"error": "need at least 2 groups with >= 2 observations"}
    stat, p = stats.f_oneway(*groups)
    return {
        "test": "one-way ANOVA",
        "n_groups": len(groups),
        "f_statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant_at_0.05": bool(p < 0.05),
    }


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-square test of independence between two categorical columns."""
    table = pd.crosstab(df[col_a], df[col_b])
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return {"error": "contingency table too small for chi-square"}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "test": "chi-square independence",
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": round(float(p), 6),
        "significant_at_0.05": bool(p < 0.05),
    }
