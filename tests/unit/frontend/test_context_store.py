"""Frontend context service tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iad.state.session import (
    KEY_DATASET,
    KEY_MODEL_BUNDLE,
    KEY_TRAINING_REPORT,
    init_session_state,
    state_get,
    state_set,
)


@pytest.fixture(autouse=True)
def _session():
    init_session_state()
    yield


@pytest.mark.unit
def test_require_dataframe_missing(monkeypatch) -> None:
    from iad.frontend.services import context

    monkeypatch.setattr(context.alerts, "no_dataset_hint", MagicMock())
    assert context.require_dataframe("test page") is None


@pytest.mark.unit
def test_require_dataframe_present(iris_df) -> None:
    from iad.frontend.services import context

    state_set(KEY_DATASET, iris_df)
    df = context.require_dataframe()
    assert df is not None
    assert len(df) == len(iris_df)


@pytest.mark.unit
def test_store_dataset_resets_model(iris_df) -> None:
    from iad.frontend.services.context import store_dataset

    state_set(KEY_MODEL_BUNDLE, {"model": 1})
    state_set(KEY_TRAINING_REPORT, {"report": 1})
    store_dataset(iris_df, "iris", suggested_target="species")
    assert state_get(KEY_MODEL_BUNDLE) is None
    assert state_get(KEY_TRAINING_REPORT) is None
