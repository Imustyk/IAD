"""Session-state accessors — backward-compatible keys + typed snapshots."""
from __future__ import annotations

import pytest

import iad.state.session as session
from iad.state.session import (
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TASK_TYPE,
    SessionContext,
    init_session_state,
    state_get,
    state_set,
)


@pytest.fixture
def mock_state(monkeypatch: pytest.MonkeyPatch) -> dict:
    fake: dict = {}
    monkeypatch.setattr(session, "_streamlit_state", lambda: fake)
    return fake


def test_init_populates_defaults(mock_state: dict) -> None:
    init_session_state()
    assert KEY_DATASET in mock_state
    assert mock_state[KEY_DATASET] is None
    assert "title" in mock_state["business_case"]
    assert mock_state["request_id"]


def test_init_is_idempotent(mock_state: dict) -> None:
    init_session_state()
    mock_state[KEY_DATASET] = "preserved"
    init_session_state()
    assert mock_state[KEY_DATASET] == "preserved"


def test_get_and_set(mock_state: dict) -> None:
    state_set("custom_key", 42)
    assert state_get("custom_key") == 42
    assert state_get("missing", "default") == "default"


def test_session_context_snapshot(mock_state: dict) -> None:
    init_session_state()
    state_set(KEY_DATASET, object())  # truthy
    state_set(KEY_DATASET_NAME, "iris.csv")
    state_set(KEY_TARGET_COLUMN, "species")
    state_set(KEY_TASK_TYPE, "classification")
    state_set(KEY_MODEL_BUNDLE, "fake-pipeline")

    ctx = SessionContext.current()
    assert ctx.dataset_name == "iris.csv"
    assert ctx.target_column == "species"
    assert ctx.task_type == "classification"
    assert ctx.has_dataset is True
    assert ctx.has_model is True

    extra = ctx.to_log_extra()
    assert extra["ctx_dataset"] == "iris.csv"
    assert extra["ctx_task"] == "classification"
    assert extra["ctx_has_model"] is True


def test_session_context_empty(mock_state: dict) -> None:
    ctx = SessionContext.current()
    assert ctx.dataset_name is None
    assert ctx.has_dataset is False
    assert ctx.has_model is False


def test_keys_match_legacy_src_utils() -> None:
    """Critical backward-compat invariant: the new session keys must match
    the legacy ``src.utils.SESSION_KEYS`` values verbatim."""
    from src.utils import SESSION_KEYS

    assert SESSION_KEYS["dataset"] == KEY_DATASET
    assert SESSION_KEYS["dataset_name"] == KEY_DATASET_NAME
    assert SESSION_KEYS["target_column"] == KEY_TARGET_COLUMN
    assert SESSION_KEYS["task_type"] == KEY_TASK_TYPE
    assert SESSION_KEYS["model_bundle"] == KEY_MODEL_BUNDLE
