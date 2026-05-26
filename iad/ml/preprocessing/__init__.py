"""Phase 2 — Data engineering layer.

Public API
==========

Schemas (Pandera + GE adapter)::

    from iad.ml.preprocessing import (
        SchemaValidationResult,
        validate_with_pandera,
        SAMPLE_SCHEMAS,
        get_sample_schema,
    )

Quality checks::

    from iad.ml.preprocessing import (
        detect_duplicates,
        null_report,
        detect_outliers_iqr,
        detect_outliers_isolation_forest,
        ImpossibleValueRule,
        check_impossible_values,
    )

Drift detection::

    from iad.ml.preprocessing import DriftDetector, DriftReport

Profiling::

    from iad.ml.preprocessing import DataProfiler

sklearn-compatible transformers::

    from iad.ml.preprocessing import (
        DatetimeFeatureExtractor,
        RareCategoryGrouper,
        SkewnessCorrector,
        MulticollinearityReducer,
        SmoothedTargetEncoder,
        AutoFeatureSelector,
    )

Pipeline composition::

    from iad.ml.preprocessing import (
        PreprocessingPipelineBuilder,
        build_auto_preprocessor,
    )

Exceptions::

    from iad.ml.preprocessing import PreprocessingError, SchemaValidationFailed
"""
from __future__ import annotations

from iad.ml.preprocessing.drift import (
    ColumnDriftResult,
    DriftDetector,
    DriftReport,
)
from iad.ml.preprocessing.exceptions import (
    DriftDetectionError,
    PreprocessingError,
    SchemaValidationFailed,
    TransformerNotFittedError,
)
from iad.ml.preprocessing.pipelines import (
    PreprocessingPipelineBuilder,
    build_auto_preprocessor,
)
from iad.ml.preprocessing.profiling import DataProfile, DataProfiler
from iad.ml.preprocessing.quality import (
    DuplicateReport,
    ImpossibleValueReport,
    ImpossibleValueRule,
    NullReport,
    OutlierReport,
    check_impossible_values,
    columns_above_null_threshold,
    detect_duplicates,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    null_report,
)
from iad.ml.preprocessing.schemas import (
    SAMPLE_SCHEMAS,
    SchemaValidationResult,
    coerce_with_schema,
    get_sample_schema,
    validate_with_pandera,
)
from iad.ml.preprocessing.transformers import (
    AutoFeatureSelector,
    DatetimeFeatureExtractor,
    MulticollinearityReducer,
    RareCategoryGrouper,
    SkewnessCorrector,
    SmoothedTargetEncoder,
)

__all__ = [
    # Schemas
    "SchemaValidationResult",
    "validate_with_pandera",
    "coerce_with_schema",
    "SAMPLE_SCHEMAS",
    "get_sample_schema",
    # Quality
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
    # Drift
    "DriftDetector",
    "DriftReport",
    "ColumnDriftResult",
    # Profiling
    "DataProfile",
    "DataProfiler",
    # Transformers
    "DatetimeFeatureExtractor",
    "RareCategoryGrouper",
    "SkewnessCorrector",
    "MulticollinearityReducer",
    "SmoothedTargetEncoder",
    "AutoFeatureSelector",
    # Pipelines
    "PreprocessingPipelineBuilder",
    "build_auto_preprocessor",
    # Exceptions
    "PreprocessingError",
    "SchemaValidationFailed",
    "DriftDetectionError",
    "TransformerNotFittedError",
]
