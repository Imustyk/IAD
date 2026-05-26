"""Phase 2 integration: combined preprocessing + drift + profile end-to-end.

These tests exercise the new data-engineering layer against real sample
datasets and assert it composes correctly with the existing Phase-0
training entrypoint in ``src.predictive``. That is the contract Phase 3
will rely on when wiring ``iad.ml.training``.
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_full_pipeline_on_telco_with_phase2_preprocessor() -> None:
    """Build the auto-preprocessor on Telco, train a small classifier, predict."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from iad.ml.preprocessing import build_auto_preprocessor
    from src.data_loader import load_sample

    df = load_sample("Telco churn (classification)")
    pre, schema = build_auto_preprocessor(df, target="churn", task="classification")

    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)),
        ]
    )
    pipe.fit(df.drop(columns=["churn"]), df["churn"])
    score = pipe.score(df.drop(columns=["churn"]), df["churn"])
    assert score > 0.8
    assert schema["numeric"]
    assert schema["categorical"]


@pytest.mark.integration
def test_drift_pipeline_iris() -> None:
    from iad.ml.preprocessing import DataProfiler, DriftDetector
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")

    profile = DataProfiler().profile(df)
    assert profile.quality_score > 0.7

    half = df.sample(frac=0.5, random_state=1).reset_index(drop=True)
    shifted = half.copy()
    shifted["sepal length (cm)"] = shifted["sepal length (cm)"] + 2.0  # synthetic drift

    detector = DriftDetector(psi_threshold=0.25).fit(df)
    report = detector.detect(shifted)
    assert report.overall_drift_detected
    flagged = {c.column for c in report.columns if c.drift_detected}
    assert "sepal length (cm)" in flagged


@pytest.mark.integration
def test_phase2_does_not_break_legacy_training() -> None:
    """Phase 2 must not regress Phase 1's legacy training pipeline."""
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
    assert report.metrics["accuracy"] > 0.85
    preds = predict(pipeline, df.head(5), report.features, "classification")
    assert "prediction" in preds.columns
