"""AutoML adapters."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.automl import FLAMLBackend, PyCaretBackend, flaml_available, pycaret_available


@pytest.mark.skipif(not flaml_available(), reason="flaml not installed")
def test_flaml_classification_runs() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    backend = FLAMLBackend(estimator_list=["lgbm", "rf"])
    result = backend.fit(
        df.drop(columns=["species"]),
        df["species"],
        task="classification",
        time_budget=10,
        metric="macro_f1",
    )
    assert result.backend == "flaml"
    assert result.best_model is not None
    assert result.elapsed_seconds > 0


def test_pycaret_unavailable_raises_clear_error() -> None:
    if pycaret_available():
        pytest.skip("pycaret installed; cannot test the unavailable path")
    backend = PyCaretBackend()
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    with pytest.raises(Exception) as exc:
        backend.fit(df.drop(columns=["b"]), df["b"], task="classification")
    assert "pycaret" in str(exc.value).lower()
