"""Preprocessing pipeline builder tests."""
from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from iad.ml.preprocessing._dtypes import categorical_columns, numeric_columns
from iad.ml.preprocessing.pipelines.builder import PreprocessingPipelineBuilder


@pytest.mark.unit
def test_builder_produces_pipeline(iris_df) -> None:
    target = "species"
    features = [c for c in iris_df.columns if c != target]
    X = iris_df[features]
    y = iris_df[target]

    pipe = (
        PreprocessingPipelineBuilder()
        .with_columns(
            numeric=numeric_columns(iris_df[features]),
            categorical=categorical_columns(iris_df[features]),
        )
        .with_imputation()
        .with_scaling("standard")
        .with_encoding("onehot")
        .build()
    )
    assert isinstance(pipe, Pipeline)
    transformed = pipe.fit_transform(X, y)
    assert transformed.shape[0] == len(iris_df)
