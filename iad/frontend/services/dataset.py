"""Dataset loading service — performance-aware wrapper over legacy loaders."""
from __future__ import annotations

import pandas as pd

from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.performance.lazy import LazyDatasetView
from iad.performance.loader import load_uploaded_file_fast, loading_metadata
from iad.performance.memory import prepare_for_session
from iad.state.session import (
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TRAINING_REPORT,
    state_set,
)

logger = get_logger("iad.frontend.dataset")


def store_dataset(
    df: pd.DataFrame,
    name: str,
    *,
    suggested_target: str | None = None,
    use_performance_pipeline: bool = True,
) -> LazyDatasetView:
    """Persist dataset to session with optional memory optimisation."""
    settings = get_settings()
    if use_performance_pipeline and settings.PERF_AUTO_OPTIMIZE_DTYPES:
        df = prepare_for_session(df)
    state_set(KEY_DATASET, df)
    state_set(KEY_DATASET_NAME, name)
    state_set(KEY_MODEL_BUNDLE, None)
    state_set(KEY_TRAINING_REPORT, None)
    if suggested_target is not None:
        state_set(KEY_TARGET_COLUMN, suggested_target)
    meta = loading_metadata(df)
    logger.info("dataset stored", extra={"dataset_name": name, **meta})
    return LazyDatasetView.from_dataframe(df)


def load_uploaded(uploaded: object, name: str | None = None) -> LazyDatasetView:
    """Load Streamlit upload via Polars/pandas fast path."""
    df = load_uploaded_file_fast(uploaded)
    return store_dataset(df, name or getattr(uploaded, "name", "upload"))


def load_from_legacy_loader(loader_fn: object, name: str, **kwargs: object) -> LazyDatasetView:
    """Wrap an existing ``src.data_loader`` callable with prepare_for_session."""
    df = loader_fn(**kwargs)  # type: ignore[operator]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("loader must return a pandas DataFrame")
    return store_dataset(df, name, use_performance_pipeline=True)
