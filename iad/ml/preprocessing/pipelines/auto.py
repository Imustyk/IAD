"""Inspect a DataFrame and assemble a sensible preprocessing pipeline.

This is the function ``iad.ml.training`` will call in Phase 3 to replace the
hard-coded ColumnTransformer in ``src.predictive``. It encodes a defensible
set of defaults but stays overridable through the explicit
:class:`PreprocessingPipelineBuilder`.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from iad.core.logging import get_logger
from iad.ml.preprocessing._dtypes import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
)
from iad.ml.preprocessing.pipelines.builder import PreprocessingPipelineBuilder

logger = get_logger("iad.ml.preprocessing.auto")


HIGH_CARDINALITY_THRESHOLD = 50  # categories
TARGET_ENCODING_THRESHOLD = 30   # categories that warrant target encoding


def build_auto_preprocessor(
    df: pd.DataFrame,
    target: str | None = None,
    task: Literal["classification", "regression"] = "classification",
    *,
    rare_min_frequency: float = 0.01,
    skew_threshold: float = 1.0,
    correlation_threshold: float = 0.95,
    feature_selection: bool = False,
    max_features: int | None = None,
) -> tuple[Pipeline, dict[str, list[str]]]:
    """Inspect ``df`` and build a preprocessing ``Pipeline``.

    Returns:
        Tuple of:
            * the built :class:`sklearn.pipeline.Pipeline` (unfitted),
            * a dict mapping ``{"numeric": [...], "categorical": [...],
              "datetime": [...], "target_encoded": [...]}`` so the caller
              can persist the schema alongside the trained model.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_auto_preprocessor requires a pandas DataFrame")
    if df.empty:
        raise ValueError("Cannot build a preprocessor for an empty DataFrame")

    feature_df = df.drop(columns=[target]) if target and target in df.columns else df

    numeric_cols = numeric_columns(feature_df)
    datetime_cols = datetime_columns(feature_df)
    raw_categorical = categorical_columns(feature_df)
    high_card_cats = [
        c for c in raw_categorical if feature_df[c].nunique(dropna=True) > TARGET_ENCODING_THRESHOLD
    ]
    target_encoded = high_card_cats if (target and target in df.columns) else []
    standard_cats = [c for c in raw_categorical if c not in target_encoded]

    builder = PreprocessingPipelineBuilder()
    if datetime_cols:
        builder = builder.with_datetime_features(datetime_cols, drop_original=True)
    if standard_cats:
        builder = builder.with_rare_category_grouping(
            columns=standard_cats, min_frequency=rare_min_frequency
        )
    if numeric_cols:
        builder = builder.with_skewness_correction(columns=numeric_cols, skew_threshold=skew_threshold)
    if target_encoded and target and target in df.columns:
        builder = builder.with_target_encoding(columns=target_encoded)
    if numeric_cols:
        builder = builder.with_multicollinearity_reduction(threshold=correlation_threshold)
    if feature_selection:
        builder = builder.with_feature_selection(
            task=task, max_features=max_features, correlation_threshold=correlation_threshold
        )
    # Note: we deliberately DO NOT pin column lists here. Earlier transformers
    # in the pipeline (MulticollinearityReducer, AutoFeatureSelector, the
    # DatetimeFeatureExtractor) mutate the column set, so the final
    # ColumnTransformer must re-detect numeric vs categorical at fit time.
    # The builder uses callable selectors backed by ``iad.ml.preprocessing
    # ._dtypes`` which honour the full set of categorical-like dtypes.
    pipeline = (
        builder.with_imputation()
        .with_scaling("standard")
        .with_encoding("onehot")
        .build()
    )

    schema_groups = {
        "numeric": numeric_cols,
        "categorical": standard_cats,
        "datetime": datetime_cols,
        "target_encoded": target_encoded,
    }
    logger.info(
        "auto preprocessor built",
        extra={
            "ctx_n_numeric": len(numeric_cols),
            "ctx_n_categorical": len(standard_cats),
            "ctx_n_datetime": len(datetime_cols),
            "ctx_n_target_encoded": len(target_encoded),
            "ctx_feature_selection": feature_selection,
        },
    )
    return pipeline, schema_groups
