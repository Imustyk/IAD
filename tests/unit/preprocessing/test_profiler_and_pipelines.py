"""Profiler + pipeline builder/auto."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from iad.ml.preprocessing import (
    DataProfiler,
    PreprocessingPipelineBuilder,
    build_auto_preprocessor,
)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
def test_profiler_returns_quality_score_in_unit_interval() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    profile = DataProfiler().profile(df)
    assert 0.0 <= profile.quality_score <= 1.0
    assert profile.n_rows == len(df)


def test_profiler_flags_obvious_issues() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, None, None],  # 40% null
            "b": [1, 1, 1, 1, 1],               # constant
            "c": [1, 2, 3, 4, 5],
        }
    )
    profile = DataProfiler().profile(df)
    issues_str = " ".join(profile.issues).lower()
    assert "constant" in issues_str  # caught the constant column
    assert profile.constant_columns == ["b"]
    assert profile.missing_share > 0
    assert profile.quality_score < 1.0


def test_profiler_detects_duplicate_rows() -> None:
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]}
    )
    profile = DataProfiler().profile(df)
    assert profile.duplicate_rows >= 1
    assert any("duplicate" in i.lower() for i in profile.issues)


def test_profiler_html_render_contains_score_and_table() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    profiler = DataProfiler()
    profile = profiler.profile(df)
    html = profiler.to_html(profile, title="Demo profile")
    assert "Demo profile" in html
    assert "Quality score" in html
    assert "<table" in html


def test_profiler_to_html_file_writes(tmp_path) -> None:
    profile = DataProfiler().profile(pd.DataFrame({"x": [1, 2, 3]}))
    path = DataProfiler().to_html_file(profile, tmp_path / "report.html")
    assert path.exists()
    assert "Quality score" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def test_builder_produces_sklearn_pipeline() -> None:
    pipe = (
        PreprocessingPipelineBuilder()
        .with_imputation()
        .with_scaling("standard")
        .with_encoding("onehot")
        .build()
    )
    assert isinstance(pipe, Pipeline)


def test_builder_full_chain_fits_and_transforms() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": rng.exponential(scale=2.0, size=300),
            "cat": rng.choice(["A", "B", "C", "rare"], p=[0.4, 0.4, 0.18, 0.02], size=300),
            "ts": pd.to_datetime(rng.choice(pd.date_range("2024-01-01", periods=365), size=300)),
        }
    )
    pipe = (
        PreprocessingPipelineBuilder()
        .with_datetime_features(["ts"])
        .with_rare_category_grouping(min_frequency=0.05)
        .with_skewness_correction(skew_threshold=0.5)
        .with_imputation()
        .with_scaling("standard")
        .with_encoding("onehot")
        .build()
    )
    transformed = pipe.fit_transform(df)
    assert transformed.shape[0] == df.shape[0]
    assert transformed.shape[1] > 1


def test_builder_disable_column_transformer() -> None:
    pipe = (
        PreprocessingPipelineBuilder()
        .with_rare_category_grouping(min_frequency=0.05)
        .disable_column_transformer()
        .build()
    )
    assert isinstance(pipe, Pipeline)


# ---------------------------------------------------------------------------
# Auto-preprocessor
# ---------------------------------------------------------------------------
def test_auto_preprocessor_iris() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    pipe, schema = build_auto_preprocessor(df, target="species", task="classification")
    transformed = pipe.fit_transform(df.drop(columns=["species"]), df["species"])
    assert transformed.shape[0] == len(df)
    assert "species" not in schema["categorical"]
    assert len(schema["numeric"]) >= 4


def test_auto_preprocessor_telco() -> None:
    from src.data_loader import load_sample

    df = load_sample("Telco churn (classification)")
    pipe, schema = build_auto_preprocessor(df, target="churn", task="classification")
    transformed = pipe.fit_transform(df.drop(columns=["churn"]), df["churn"])
    assert transformed.shape[0] == len(df)
    assert "contract_type" in schema["categorical"]


def test_auto_preprocessor_handles_no_target() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    pipe, schema = build_auto_preprocessor(df, target=None, task="classification")
    transformed = pipe.fit_transform(df)
    assert transformed.shape[0] == len(df)
    assert schema["target_encoded"] == []
