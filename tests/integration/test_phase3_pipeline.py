"""Phase 3 integration: training service + tuning + explainability + tracking + persistence."""
from __future__ import annotations

import pytest

from iad.ml.explainability import SHAPExplainer
from iad.ml.tracking import MLflowTracker, mlflow_available
from iad.ml.training import (
    TrainingConfig,
    TrainingService,
    load_bundle,
    save_bundle,
)
from iad.ml.tuning import OptunaSearch


@pytest.mark.integration
def test_full_phase3_workflow_telco(tmp_path) -> None:
    """End-to-end: load data → train leaderboard → tune best → explain → persist."""
    from src.data_loader import load_sample

    df = load_sample("Telco churn (classification)")

    # 1. Train a small leaderboard so the test runs fast.
    service = TrainingService()
    config = TrainingConfig(
        task="classification",
        cv_folds=3,
        selected_models=["Logistic Regression", "Random Forest", "LightGBM"],
        cross_validate_best=True,
    )
    result = service.train(df, target="churn", config=config)
    assert result.best_entry.metrics["accuracy"] > 0.7
    assert result.model_card is not None
    leaderboard = result.leaderboard_frame()
    assert "model" in leaderboard.columns
    assert len(leaderboard) >= 3

    # 2. Hyperparameter tuning of the same family on a tiny budget.
    X = df.drop(columns=["churn"])
    y = df["churn"]
    search = OptunaSearch(
        model_name="Random Forest",
        task="classification",
        cv=3,
        n_trials=5,
        random_state=0,
    )
    tuned = search.fit(X, y)
    assert tuned.n_trials >= 1
    assert tuned.best_estimator is not None

    # 3. SHAP explanation on the tuned model.
    expl = SHAPExplainer.from_pipeline(tuned.best_estimator, X.sample(60, random_state=0), max_background_rows=30)
    importance = expl.global_importance(X.sample(40, random_state=1))
    assert importance["mean_abs_shap"].sum() > 0

    # 4. Persist + reload the original best pipeline & card.
    bundle_path = tmp_path / "telco_best.joblib"
    save_bundle(result.best_pipeline, result.model_card, path=bundle_path)
    pipeline_loaded, card_loaded = load_bundle(bundle_path, trusted_path_only=False)
    sample_predictions = pipeline_loaded.predict(X.head(5))
    assert len(sample_predictions) == 5
    assert card_loaded.dataset_fingerprint == result.model_card.dataset_fingerprint


@pytest.mark.integration
@pytest.mark.skipif(not mlflow_available(), reason="mlflow not installed")
def test_training_with_mlflow(tmp_path, monkeypatch) -> None:
    """The TrainingService path with track_mlflow=True must not crash."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    config = TrainingConfig(
        task="classification",
        selected_models=["Logistic Regression"],
        cv_folds=3,
        cross_validate_best=False,
        track_mlflow=True,
        mlflow_experiment="iad_phase3_test",
    )
    service = TrainingService()
    # Manually open a tracker to confirm mlflow can write to tmp_path.
    with MLflowTracker(experiment="iad_phase3_test", tracking_uri=str(tmp_path / "mlruns")) as tracker:
        tracker.log_params({"smoke": True})
    result = service.train(df, target="species", config=config)
    assert result.best_entry.metrics["accuracy"] > 0.85


@pytest.mark.integration
def test_phase3_does_not_break_legacy_training() -> None:
    """Phase 3 must not regress the Phase 1/2 legacy training pipeline."""
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
