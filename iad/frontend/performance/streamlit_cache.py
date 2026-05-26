"""Centralised Streamlit caching with settings-driven TTL."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
import streamlit as st

from iad.config.settings import get_settings
from iad.performance.fingerprints import dataframe_fingerprint, params_fingerprint

F = TypeVar("F", bound=Callable[..., Any])


def cache_ttl() -> int:
    return get_settings().UI_CACHE_TTL_SECONDS


def cache_dataframe(**st_kwargs: Any) -> Callable[[F], F]:
    """Decorator for functions that return a DataFrame keyed by fingerprint."""

    def decorator(fn: F) -> F:
        ttl = st_kwargs.pop("ttl", cache_ttl())

        @st.cache_data(show_spinner=st_kwargs.pop("show_spinner", False), ttl=ttl, **st_kwargs)
        def wrapper(fp: str, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        def call(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
            fp = dataframe_fingerprint(df)
            return wrapper(fp, *args, **kwargs)

        return call  # type: ignore[return-value]

    return decorator


@st.cache_data(show_spinner=False, ttl=300)
def cached_correlation_matrix(
    fp: str,
    corr_values: tuple[tuple[float, ...], ...],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Rehydrate a correlation matrix from hashed tuple data."""
    return pd.DataFrame(corr_values, index=list(columns), columns=list(columns))


def get_or_compute_correlation(
    df: pd.DataFrame,
    method: str,
    compute_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Cache correlation heatmap data by dataframe fingerprint + method."""
    fp = dataframe_fingerprint(df, extra=method)
    corr = compute_fn()
    return cached_correlation_matrix(
        fp,
        tuple(tuple(row) for row in corr.values.tolist()),
        tuple(corr.columns.astype(str)),
    )


@st.cache_resource(show_spinner=False)
def cached_model_bundle(bundle_bytes: bytes, bundle_id: str) -> object:
    """Cache loaded joblib bundles by content hash id."""
    import io

    import joblib

    return joblib.load(io.BytesIO(bundle_bytes))


@st.cache_data(show_spinner=False, ttl=120)
def cached_descriptive_stats(fp: str, stats_json: str) -> pd.DataFrame:
    """Cache serialised descriptive statistics."""
    from io import StringIO

    return pd.read_json(StringIO(stats_json))


def cache_by_params(fn: F) -> F:
    """Cache function output keyed by hashed kwargs."""
    ttl = cache_ttl()

    @st.cache_data(show_spinner=False, ttl=ttl)
    def wrapper(param_key: str, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def call(*args: Any, **kwargs: Any) -> Any:
        key = params_fingerprint(args=list(args), **kwargs)
        return wrapper(key, *args, **kwargs)

    return call  # type: ignore[return-value]
