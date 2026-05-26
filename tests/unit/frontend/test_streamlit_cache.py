"""Streamlit cache helper tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_cache_ttl(settings) -> None:
    from iad.frontend.performance.streamlit_cache import cache_ttl

    assert cache_ttl() == settings.UI_CACHE_TTL_SECONDS


@pytest.mark.unit
def test_cached_correlation_matrix(iris_df, monkeypatch) -> None:
    from iad.frontend.performance import streamlit_cache

    def passthrough_cache(**kwargs):
        def decorator(fn):
            return fn

        return decorator

    mock_st = MagicMock()
    mock_st.cache_data = passthrough_cache
    mock_st.cache_resource = passthrough_cache
    monkeypatch.setattr(streamlit_cache, "st", mock_st)

    numeric = iris_df.select_dtypes(include="number")
    corr = numeric.corr()
    fp = "test-fp"
    restored = streamlit_cache.cached_correlation_matrix(
        fp,
        tuple(tuple(row) for row in corr.values.tolist()),
        tuple(corr.columns.astype(str)),
    )
    assert restored.shape == corr.shape

    out = streamlit_cache.get_or_compute_correlation(iris_df, "pearson", lambda: corr)
    assert out.shape == corr.shape


@pytest.mark.unit
def test_cache_by_params(monkeypatch) -> None:
    from iad.frontend.performance import streamlit_cache

    def passthrough_cache(**kwargs):
        def decorator(fn):
            return fn

        return decorator

    mock_st = MagicMock()
    mock_st.cache_data = passthrough_cache
    monkeypatch.setattr(streamlit_cache, "st", mock_st)

    @streamlit_cache.cache_by_params
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
