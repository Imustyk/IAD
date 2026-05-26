"""Descriptive analytics: summary statistics, distributions, missing values."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed numeric summary including skewness and kurtosis."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()

    desc = numeric.describe().transpose()
    desc["missing"] = df[numeric.columns].isna().sum().values
    desc["unique"] = numeric.nunique().values
    desc["skew"] = numeric.skew(numeric_only=True).values
    desc["kurtosis"] = numeric.kurt(numeric_only=True).values
    desc["variance"] = numeric.var(numeric_only=True).values
    desc["range"] = (numeric.max() - numeric.min()).values

    columns_order = [
        "count", "missing", "unique", "mean", "std", "variance",
        "min", "25%", "50%", "75%", "max", "range", "skew", "kurtosis",
    ]
    return desc[[c for c in columns_order if c in desc.columns]].round(4)


def categorical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        series = df[col]
        counts = series.value_counts(dropna=False)
        top_value = counts.index[0] if len(counts) else np.nan
        top_freq = int(counts.iloc[0]) if len(counts) else 0
        rows.append(
            {
                "column": col,
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "top_value": top_value,
                "top_freq": top_freq,
                "top_share_%": round(top_freq / max(len(series), 1) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def value_counts_table(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).head(top_n)
    out = counts.rename_axis(column).reset_index(name="count")
    out["percent"] = (out["count"] / max(len(df), 1) * 100).round(2)
    return out


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / max(len(df), 1) * 100).round(2)
    dtypes = df.dtypes.astype(str)
    out = pd.DataFrame(
        {
            "column": missing.index,
            "missing": missing.values,
            "percent": pct.values,
            "dtype": dtypes.values,
        }
    )
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def distribution_normality(df: pd.DataFrame, column: str) -> dict:
    """Shapiro-Wilk or D'Agostino test for approximate normality."""
    series = df[column].dropna()
    if len(series) < 8:
        return {"test": "n/a", "p_value": np.nan, "is_normal": False, "n": int(len(series))}
    try:
        if len(series) <= 5000:
            stat, p_value = stats.shapiro(series)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = stats.normaltest(series)
            test_name = "D'Agostino K^2"
    except Exception:
        return {"test": "n/a", "p_value": np.nan, "is_normal": False, "n": int(len(series))}
    return {
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "is_normal": bool(p_value > 0.05),
        "n": int(len(series)),
    }


def time_series_summary(df: pd.DataFrame, datetime_col: str, value_col: str) -> pd.DataFrame:
    """Resample a numeric column by month and return mean/sum/count."""
    s = df[[datetime_col, value_col]].dropna().copy()
    if s.empty:
        return pd.DataFrame()
    s[datetime_col] = pd.to_datetime(s[datetime_col], errors="coerce")
    s = s.dropna(subset=[datetime_col]).set_index(datetime_col).sort_index()
    if s.empty:
        return pd.DataFrame()
    grouped = s[value_col].resample("MS").agg(["mean", "sum", "count"]).reset_index()
    grouped.columns = [datetime_col, "mean", "sum", "count"]
    return grouped
