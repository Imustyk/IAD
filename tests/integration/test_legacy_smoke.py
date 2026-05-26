"""Backward-compatibility regression tests.

Phase 1 must not break the existing ``src``/``pages`` pipeline. These tests
exercise the legacy public APIs end-to-end with a small sample dataset.
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_legacy_src_imports_clean() -> None:
    from src import (  # noqa: F401
        data_loader,
        descriptive,
        diagnostic,
        predictive,
        prescriptive,
        utils,
    )

    assert hasattr(data_loader, "SAMPLE_DATASETS")
    assert hasattr(predictive, "train_models")


@pytest.mark.integration
def test_sample_dataset_loads() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    assert not df.empty
    assert "species" in df.columns


@pytest.mark.integration
def test_full_training_pipeline_parity() -> None:
    """End-to-end training on Iris must still succeed and meet a sane bar."""
    from src.data_loader import load_sample
    from src.predictive import predict, train_models

    df = load_sample("Iris (classification)")
    pipeline, report = train_models(
        df=df,
        target="species",
        feature_columns=[c for c in df.columns if c != "species"],
        task_type="classification",
        cv_folds=3,
    )

    assert report.metrics["accuracy"] > 0.85, report.metrics
    preds = predict(pipeline, df.head(5), report.features, "classification")
    assert "prediction" in preds.columns
    assert len(preds) == 5
