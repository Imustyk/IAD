"""Data-quality checks: duplicates, nulls, outliers, impossible-value rules."""
from iad.ml.preprocessing.quality.duplicates import DuplicateReport, detect_duplicates
from iad.ml.preprocessing.quality.nulls import NullReport, columns_above_null_threshold, null_report
from iad.ml.preprocessing.quality.outliers import (
    OutlierReport,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
)
from iad.ml.preprocessing.quality.rules import (
    ImpossibleValueReport,
    ImpossibleValueRule,
    check_impossible_values,
)

__all__ = [
    "DuplicateReport",
    "detect_duplicates",
    "NullReport",
    "null_report",
    "columns_above_null_threshold",
    "OutlierReport",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_isolation_forest",
    "ImpossibleValueRule",
    "ImpossibleValueReport",
    "check_impossible_values",
]
