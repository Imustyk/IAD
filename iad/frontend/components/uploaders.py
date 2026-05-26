"""File upload and URL loading UI components."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from iad.backend.security.upload_policy import validate_upload
from iad.config.settings import get_settings
from iad.core.exceptions import UploadError
from iad.frontend.components import alerts as alert_ui


def file_uploader(
    *,
    label: str = "Upload a data file",
    key: str | None = None,
    on_success: Callable[[pd.DataFrame, str], None] | None = None,
    loader: Callable[[object], pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Render a validated file uploader.

    Parameters
    ----------
    loader:
        Callable that accepts a Streamlit ``UploadedFile`` and returns a
        DataFrame (typically ``src.data_loader.load_uploaded_file``).
    on_success:
        Callback ``(df, filename)`` invoked after a successful load.
    """
    settings = get_settings()
    extensions = [ext.lstrip(".") for ext in settings.ALLOWED_FILE_EXTENSIONS]

    uploaded = st.file_uploader(
        label,
        type=extensions,
        accept_multiple_files=False,
        key=key,
        help=f"Max size: {settings.MAX_UPLOAD_MB} MB. Allowed: {', '.join(extensions)}",
    )
    if uploaded is None:
        return None

    try:
        validate_upload(
            filename=uploaded.name,
            size_bytes=uploaded.size,
            content_type=getattr(uploaded, "type", None),
            settings=settings,
        )
    except UploadError as exc:
        alert_ui.error(exc.user_message, show_details=True, exc=exc)
        return None

    if loader is None:
        alert_ui.warning("No loader configured for uploads.")
        return None

    try:
        df = loader(uploaded)
    except Exception as exc:
        alert_ui.error(f"Could not read file: {exc}", show_details=True, exc=exc)
        return None

    if len(df) > settings.MAX_ROWS:
        alert_ui.warning(
            f"Dataset has {len(df):,} rows (limit {settings.MAX_ROWS:,}). "
            "Consider sampling or using Polars/Dask in a future release."
        )
    if df.shape[1] > settings.MAX_COLUMNS:
        alert_ui.warning(f"Dataset has {df.shape[1]:,} columns (limit {settings.MAX_COLUMNS:,}).")

    if on_success is not None:
        on_success(df, uploaded.name)
    return df


def url_loader(
    *,
    placeholder: str = "https://example.com/data.csv",
    key: str = "url_input",
    button_label: str = "Load from URL",
    loader: Callable[[str], pd.DataFrame] | None = None,
    on_success: Callable[[pd.DataFrame, str], None] | None = None,
) -> pd.DataFrame | None:
    """Render URL input + load button."""
    url = st.text_input("Dataset URL", placeholder=placeholder, key=key)
    if not st.button(button_label, type="primary"):
        return None
    if not url.strip():
        alert_ui.warning("Enter a URL first.")
        return None
    if loader is None:
        alert_ui.warning("No URL loader configured.")
        return None
    try:
        df = loader(url.strip())
    except Exception as exc:
        alert_ui.error(f"Could not load URL: {exc}", show_details=True, exc=exc)
        return None
    if on_success is not None:
        on_success(df, url.strip())
    return df


def sample_dataset_selector(
    sample_names: list[str],
    descriptions: dict[str, str] | None = None,
    *,
    default_index: int = 0,
    key: str = "sample_select",
) -> str | None:
    """Render sample dataset picker; returns selected name when Load clicked."""
    name = st.selectbox("Sample dataset", sample_names, index=default_index, key=key)
    if descriptions and name in descriptions:
        st.info(descriptions[name])
    return name
