"""Fluent builder for sklearn preprocessing pipelines.

Why a builder?
    The legacy ``src.predictive.build_preprocessor`` hard-codes its column
    transformer. Real use-cases need more flexibility: some datasets need
    target encoding, some need datetime extraction, some need outlier
    capping. A fluent builder lets services compose what they need without
    coupling to a specific data shape.

Usage::

    pipe = (
        PreprocessingPipelineBuilder()
            .with_datetime_features(["signup_date"])
            .with_rare_category_grouping(min_frequency=0.005)
            .with_skewness_correction(skew_threshold=1.0)
            .with_imputation()
            .with_scaling()
            .with_encoding()
            .build()
    )
"""
from __future__ import annotations

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from iad.ml.preprocessing.transformers import (
    AutoFeatureSelector,
    DatetimeFeatureExtractor,
    MulticollinearityReducer,
    RareCategoryGrouper,
    SkewnessCorrector,
    SmoothedTargetEncoder,
)

ScalerKind = Literal["standard", "minmax", "robust", "none"]
EncoderKind = Literal["onehot", "ordinal"]


class PreprocessingPipelineBuilder:
    """Stage-by-stage construction of a sklearn ``Pipeline``."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, object]] = []
        self._numeric_strategy: str = "median"
        self._categorical_strategy: str = "most_frequent"
        self._scaler: ScalerKind = "standard"
        self._encoder: EncoderKind = "onehot"
        self._numeric_columns: list[str] | None = None
        self._categorical_columns: list[str] | None = None
        self._needs_column_transformer: bool = True

    # ------------------------------------------------------------------
    # Optional, ordered pre-stages
    # ------------------------------------------------------------------
    def with_datetime_features(
        self, columns: list[str] | None = None, *, drop_original: bool = True
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            ("datetime_features", DatetimeFeatureExtractor(columns=columns, drop_original=drop_original))
        )
        return self

    def with_rare_category_grouping(
        self,
        columns: list[str] | None = None,
        *,
        min_frequency: float = 0.01,
        max_categories: int | None = None,
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            (
                "rare_category",
                RareCategoryGrouper(
                    columns=columns,
                    min_frequency=min_frequency,
                    max_categories=max_categories,
                ),
            )
        )
        return self

    def with_target_encoding(
        self,
        columns: list[str] | None = None,
        *,
        smoothing: float = 10.0,
        n_folds: int = 5,
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            (
                "target_encoding",
                SmoothedTargetEncoder(columns=columns, smoothing=smoothing, n_folds=n_folds),
            )
        )
        return self

    def with_skewness_correction(
        self,
        columns: list[str] | None = None,
        *,
        skew_threshold: float = 1.0,
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            (
                "skewness",
                SkewnessCorrector(columns=columns, skew_threshold=skew_threshold),
            )
        )
        return self

    def with_multicollinearity_reduction(
        self,
        *,
        threshold: float = 0.95,
        protect: list[str] | None = None,
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            (
                "multicollinearity",
                MulticollinearityReducer(threshold=threshold, protect=protect),
            )
        )
        return self

    def with_feature_selection(
        self,
        *,
        task: Literal["classification", "regression"] = "classification",
        max_features: int | None = None,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        use_model_based: bool = True,
    ) -> PreprocessingPipelineBuilder:
        self._steps.append(
            (
                "feature_selection",
                AutoFeatureSelector(
                    task=task,
                    max_features=max_features,
                    variance_threshold=variance_threshold,
                    correlation_threshold=correlation_threshold,
                    use_model_based=use_model_based,
                ),
            )
        )
        return self

    # ------------------------------------------------------------------
    # Required impute / scale / encode stage
    # ------------------------------------------------------------------
    def with_imputation(
        self,
        *,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ) -> PreprocessingPipelineBuilder:
        self._numeric_strategy = numeric_strategy
        self._categorical_strategy = categorical_strategy
        return self

    def with_scaling(self, kind: ScalerKind = "standard") -> PreprocessingPipelineBuilder:
        self._scaler = kind
        return self

    def with_encoding(self, kind: EncoderKind = "onehot") -> PreprocessingPipelineBuilder:
        self._encoder = kind
        return self

    def with_columns(
        self,
        *,
        numeric: list[str] | None = None,
        categorical: list[str] | None = None,
    ) -> PreprocessingPipelineBuilder:
        """Pin numeric/categorical column lists. ``None`` → infer at fit time."""
        self._numeric_columns = numeric
        self._categorical_columns = categorical
        return self

    def disable_column_transformer(self) -> PreprocessingPipelineBuilder:
        """Skip the final impute/scale/encode block (e.g. when the model
        accepts raw mixed dtypes — CatBoost, LightGBM with native categorical)."""
        self._needs_column_transformer = False
        return self

    # ------------------------------------------------------------------
    def build(self) -> Pipeline:
        steps: list[tuple[str, object]] = list(self._steps)
        if self._needs_column_transformer:
            steps.append(("column_transformer", self._build_column_transformer()))
        if not steps:
            raise ValueError("Pipeline has no steps; configure builder before calling build().")
        return Pipeline(steps=steps)

    # ------------------------------------------------------------------
    def _build_column_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self._numeric_strategy)),
                ("scaler", self._make_scaler()),
            ]
        )
        encoder = self._make_encoder()
        cat_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=self._categorical_strategy, fill_value="missing")),
        ]
        if encoder is not None:
            cat_steps.append(("encoder", encoder))
        categorical_pipeline = Pipeline(cat_steps)

        # If columns are pinned, use them directly. Otherwise rely on dtype-based selectors.
        if self._numeric_columns is not None or self._categorical_columns is not None:
            transformers = []
            if self._numeric_columns:
                transformers.append(("num", numeric_pipeline, self._numeric_columns))
            if self._categorical_columns:
                transformers.append(("cat", categorical_pipeline, self._categorical_columns))
            return ColumnTransformer(transformers=transformers, remainder="drop")

        # We use callable selectors instead of make_column_selector because
        # modern pandas reports string columns with ``dtype="str"`` which
        # ``make_column_selector(dtype_include=["object", "category"])`` does
        # not match. The ``iad.ml.preprocessing._dtypes`` helpers honour the
        # full set of categorical-like dtypes (object/string/category/bool).
        from iad.ml.preprocessing._dtypes import (
            categorical_columns as _cats,
        )
        from iad.ml.preprocessing._dtypes import (
            numeric_columns as _nums,
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, _nums),
                ("cat", categorical_pipeline, _cats),
            ],
            remainder="drop",
        )

    def _make_scaler(self):
        if self._scaler == "standard":
            return StandardScaler()
        if self._scaler == "minmax":
            return MinMaxScaler()
        if self._scaler == "robust":
            return RobustScaler()
        return "passthrough"

    def _make_encoder(self):
        if self._encoder == "onehot":
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:  # pragma: no cover — sklearn < 1.2
                return OneHotEncoder(handle_unknown="ignore", sparse=False)
        if self._encoder == "ordinal":
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return None
