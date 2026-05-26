"""Training API helpers — task inference edge cases."""
from __future__ import annotations

import pandas as pd

from iad.backend.services.training_api_service import _infer_task_type


def test_infer_task_type_string_column() -> None:
    df = pd.DataFrame({"label": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    assert _infer_task_type(df, "label") == "classification"


def test_infer_task_type_regression() -> None:
    df = pd.DataFrame({"y": [1.1, 2.2, 3.3, 4.4], "x": [1, 2, 3, 4]})
    assert _infer_task_type(df, "y") == "regression"
