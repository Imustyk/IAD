"""Unit tests for leaderboard card conversion."""
from __future__ import annotations

import pandas as pd

from iad.frontend.components.model_cards import entries_from_leaderboard_df


def test_entries_from_leaderboard_df_classification() -> None:
    df = pd.DataFrame({
        "model": ["Random Forest", "Logistic Regression"],
        "roc_auc": [0.91, 0.88],
        "family": ["ensemble", "linear"],
        "train_time_s": [1.2, 0.4],
    })
    entries = entries_from_leaderboard_df(
        df,
        task_type="classification",
        best_model_name="Random Forest",
    )
    assert len(entries) == 2
    assert entries[0].is_best is True
    assert entries[0].primary_metric == 0.91
