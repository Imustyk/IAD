"""SHAP + LIME explainers."""
from __future__ import annotations

import pytest

from iad.ml.explainability import LIMEExplainer, SHAPExplainer
from iad.ml.training import TrainingConfig, TrainingService


@pytest.fixture(scope="module")
def trained_iris():
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    config = TrainingConfig(
        task="classification",
        selected_models=["Random Forest"],
        cv_folds=3,
        cross_validate_best=False,
    )
    result = TrainingService().train(df, target="species", config=config)
    return df, result


@pytest.fixture(scope="module")
def trained_logistic():
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    config = TrainingConfig(
        task="classification",
        selected_models=["Logistic Regression"],
        cv_folds=3,
        cross_validate_best=False,
    )
    result = TrainingService().train(df, target="species", config=config)
    return df, result


def test_shap_tree_routing(trained_iris) -> None:
    df, result = trained_iris
    sample = df.drop(columns=["species"]).sample(50, random_state=0)
    expl = SHAPExplainer.from_pipeline(result.best_pipeline, sample, max_background_rows=20)
    assert expl.strategy == "tree"
    importance = expl.global_importance(sample.head(20))
    assert "feature" in importance.columns
    assert importance["mean_abs_shap"].sum() > 0


def test_shap_explain_row(trained_iris) -> None:
    df, result = trained_iris
    sample = df.drop(columns=["species"]).sample(40, random_state=1)
    expl = SHAPExplainer.from_pipeline(result.best_pipeline, sample, max_background_rows=20)
    explanation = expl.explain_row(df.drop(columns=["species"]).iloc[0])
    assert explanation.feature_names
    assert explanation.feature_values.size > 0
    waterfall = expl.waterfall_data(df.drop(columns=["species"]).iloc[0], n_top=5)
    assert {"feature", "value", "shap"}.issubset(set(waterfall.columns))


def test_shap_linear_routing(trained_logistic) -> None:
    df, result = trained_logistic
    sample = df.drop(columns=["species"]).sample(50, random_state=0)
    expl = SHAPExplainer.from_pipeline(result.best_pipeline, sample, max_background_rows=20)
    # Logistic regression should prefer the linear strategy; if SHAP's
    # LinearExplainer can't initialise, the explainer falls back to kernel.
    assert expl.strategy in {"linear", "kernel"}


def test_lime_local_explanation(trained_iris) -> None:
    df, result = trained_iris
    X = df.drop(columns=["species"])
    explainer = LIMEExplainer(
        result.best_pipeline,
        background=X.sample(50, random_state=0),
        mode="classification",
        class_names=sorted(df["species"].unique().tolist()),
    )
    explanation = explainer.explain(X.iloc[0], num_features=4, num_samples=200)
    assert len(explanation.contributions) > 0
    frame = explanation.to_frame()
    assert {"feature", "weight"}.issubset(set(frame.columns))
