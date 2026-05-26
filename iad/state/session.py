"""Typed Streamlit session-state layer.

Backward compatibility
----------------------
The string keys used here are **identical** to those in the legacy
``src.utils.SESSION_KEYS`` mapping. That means existing pages that read or
write ``st.session_state[...]`` continue to work unmodified during the
migration; the new code can simply import these constants for type safety.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # noqa: F401


# ---------------------------------------------------------------------------
# Session keys — keep in sync with legacy src.utils.SESSION_KEYS values.
# ---------------------------------------------------------------------------
KEY_DATASET = "dataset"
KEY_DATASET_NAME = "dataset_name"
KEY_BUSINESS_CASE = "business_case"
KEY_MODEL_BUNDLE = "model_bundle"
KEY_PREPROCESSOR = "preprocessor"
KEY_TARGET_COLUMN = "target_column"
KEY_TASK_TYPE = "task_type"
KEY_FEATURE_COLUMNS = "feature_columns"
KEY_TRAINING_REPORT = "training_report"
KEY_MODEL_BUNDLE_BYTES = "iad_model_bundle_bytes"

# Future use (Phases 5-7)
KEY_USER = "user"
KEY_REQUEST_ID = "request_id"


_DEFAULTS: dict[str, Any] = {
    KEY_DATASET: None,
    KEY_DATASET_NAME: None,
    KEY_BUSINESS_CASE: {
        "title": "",
        "problem": "",
        "objective": "",
        "kpis": "",
        "stakeholders": "",
        "data_sources": "",
    },
    KEY_MODEL_BUNDLE: None,
    KEY_PREPROCESSOR: None,
    KEY_TARGET_COLUMN: None,
    KEY_TASK_TYPE: None,
    KEY_FEATURE_COLUMNS: None,
    KEY_TRAINING_REPORT: None,
    KEY_MODEL_BUNDLE_BYTES: None,
    KEY_USER: None,
}


def _streamlit_state() -> Any:
    """Return ``streamlit.session_state`` lazily; useful for non-UI tests."""
    import streamlit as st  # local to keep tests independent of Streamlit

    return st.session_state


def init_session_state() -> None:
    """Populate session_state with default values; safe to call repeatedly."""
    state = _streamlit_state()
    for key, value in _DEFAULTS.items():
        if key not in state:
            state[key] = value
    if KEY_REQUEST_ID not in state:
        state[KEY_REQUEST_ID] = uuid.uuid4().hex


def state_get(key: str, default: Any = None) -> Any:
    return _streamlit_state().get(key, default)


def state_set(key: str, value: Any) -> None:
    _streamlit_state()[key] = value


@dataclass(frozen=True)
class SessionContext:
    """Immutable snapshot of the most relevant session-state values.

    Used by services and pages that need a read-only view of "current state"
    without coupling to Streamlit primitives.
    """

    dataset_name: str | None
    target_column: str | None
    task_type: str | None
    feature_columns: list[str] | None
    has_dataset: bool
    has_model: bool
    request_id: str | None

    @classmethod
    def current(cls) -> SessionContext:
        return cls(
            dataset_name=state_get(KEY_DATASET_NAME),
            target_column=state_get(KEY_TARGET_COLUMN),
            task_type=state_get(KEY_TASK_TYPE),
            feature_columns=state_get(KEY_FEATURE_COLUMNS),
            has_dataset=state_get(KEY_DATASET) is not None,
            has_model=state_get(KEY_MODEL_BUNDLE) is not None,
            request_id=state_get(KEY_REQUEST_ID),
        )

    def to_log_extra(self) -> dict[str, Any]:
        """Return a dict suitable as ``logger.info(..., extra=...)``."""
        return {
            "ctx_dataset": self.dataset_name,
            "ctx_target": self.target_column,
            "ctx_task": self.task_type,
            "ctx_has_model": self.has_model,
            "ctx_request_id": self.request_id,
        }
