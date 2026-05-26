"""Unified training report conversion tests."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iad.frontend.services.training import _result_to_unified
from iad.ml.evaluation.reports import ConfusionMatrixReport
from iad.ml.training.reports import LeaderboardEntry, TrainingResult


@pytest.mark.unit
def test_result_to_unified_classification() -> None:
    entry = LeaderboardEntry(
        model_name="Logistic Regression",
        family="linear",
        metrics={"accuracy": 0.9},
        cv_metrics={"accuracy": 0.88},
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    cm = ConfusionMatrixReport(
        labels=["a", "b"],
        matrix=pd.DataFrame([[8, 1], [1, 8]], index=["a", "b"], columns=["a", "b"]),
        per_class=pd.DataFrame(
            {"precision": [0.9, 0.9], "recall": [0.9, 0.9], "support": [9, 9]},
            index=["a", "b"],
        ),
        accuracy=0.9,
        n_samples=18,
    )
    result = TrainingResult(
        task="classification",
        target="y",
        features=["x1"],
        schema_groups={"numeric": ["x1"], "categorical": []},
        leaderboard=[entry],
        best_pipeline=pipe,
        best_entry=entry,
        test_predictions=pd.DataFrame({"y": ["a"], "y_pred": ["a"]}),
        confusion_matrix=cm,
        feature_importance=pd.DataFrame({"feature": ["x1"], "importance": [1.0]}),
    )
    unified = _result_to_unified(result)
    assert unified.engine == "enterprise"
    assert unified.confusion_matrix is not None
    assert unified.feature_importance is not None
