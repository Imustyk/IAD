"""Lazy dataset view tests."""
from __future__ import annotations

import pytest

from iad.performance.lazy import LazyDatasetView
from tests.helpers.factories import large_dataframe


@pytest.mark.unit
def test_lazy_view_small(iris_df) -> None:
    view = LazyDatasetView.from_dataframe(iris_df)
    assert view.total_rows == len(iris_df)
    assert view.was_sampled is False
    assert "rows" in view.summary_line()


@pytest.mark.unit
def test_lazy_view_page(iris_df) -> None:
    view = LazyDatasetView.from_dataframe(iris_df)
    page0 = view.page(0, page_size=10)
    assert len(page0) == 10


@pytest.mark.unit
def test_lazy_view_large_sampled(settings) -> None:
    df = large_dataframe(n=settings.PERF_LAZY_PREVIEW_ROWS + 500)
    view = LazyDatasetView.from_dataframe(df, preview_rows=100)
    assert view.was_sampled is True
    assert len(view.preview) <= 100
    assert "preview shows" in view.summary_line()
