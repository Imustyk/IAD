"""Streamlit API compatibility helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.mark.unit
def test_dataframe_omits_none_height(monkeypatch) -> None:
    from iad.frontend import streamlit_compat

    mock_st = MagicMock()
    monkeypatch.setattr(streamlit_compat, "st", mock_st)
    df = pd.DataFrame({"a": [1, 2]})

    streamlit_compat.dataframe(df)

    _, kwargs = mock_st.dataframe.call_args
    assert "height" not in kwargs


@pytest.mark.unit
def test_dataframe_passes_explicit_height(monkeypatch) -> None:
    from iad.frontend import streamlit_compat

    mock_st = MagicMock()
    monkeypatch.setattr(streamlit_compat, "st", mock_st)
    df = pd.DataFrame({"a": [1]})

    streamlit_compat.dataframe(df, height=240)

    _, kwargs = mock_st.dataframe.call_args
    assert kwargs["height"] == 240
