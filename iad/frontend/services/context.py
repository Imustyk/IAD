"""Page context helpers — dataset guards and session bridging."""
from __future__ import annotations

import pandas as pd

from iad.frontend.components import alerts
from iad.state.session import (
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_FEATURE_COLUMNS,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TASK_TYPE,
    KEY_TRAINING_REPORT,
    SessionContext,
    init_session_state,
    state_get,
    state_set,
)


def ensure_session() -> None:
    init_session_state()


def get_dataframe() -> pd.DataFrame | None:
    return state_get(KEY_DATASET)


def require_dataframe(page_name: str = "this page") -> pd.DataFrame | None:
    """Return dataset or show hint; caller should ``st.stop()`` on None."""
    df = get_dataframe()
    if df is None:
        alerts.no_dataset_hint(page_name)
        return None
    return df


def store_dataset(df: pd.DataFrame, name: str, *, suggested_target: str | None = None) -> None:
    """Persist dataset and reset downstream ML artifacts."""
    from iad.performance.memory import prepare_for_session

    df = prepare_for_session(df)
    state_set(KEY_DATASET, df)
    state_set(KEY_DATASET_NAME, name)
    state_set(KEY_MODEL_BUNDLE, None)
    state_set(KEY_TRAINING_REPORT, None)
    if suggested_target is not None:
        state_set(KEY_TARGET_COLUMN, suggested_target)


def session_context() -> SessionContext:
    return SessionContext.current()


# Re-export keys for pages migrating off src.utils
SESSION_KEYS = {
    "dataset": KEY_DATASET,
    "dataset_name": KEY_DATASET_NAME,
    "model_bundle": KEY_MODEL_BUNDLE,
    "preprocessor": "preprocessor",
    "target_column": KEY_TARGET_COLUMN,
    "task_type": KEY_TASK_TYPE,
    "feature_columns": KEY_FEATURE_COLUMNS,
    "training_report": KEY_TRAINING_REPORT,
    "business_case": "business_case",
}
