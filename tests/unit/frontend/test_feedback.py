"""Feedback helper tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_loading_context(monkeypatch) -> None:
    from iad.frontend.components import feedback

    mock_st = MagicMock()
    mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(feedback, "st", mock_st)

    with feedback.loading("Working…"):
        pass
    mock_st.spinner.assert_called_once_with("Working…")
