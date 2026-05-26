"""Table empty-state tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.mark.unit
def test_render_preview_empty(monkeypatch) -> None:
    from iad.frontend.components import tables
    import iad.frontend.components.ui as ui

    empty_mock = MagicMock()
    monkeypatch.setattr(ui, "render_html_panel", empty_mock)
    tables.render_preview(pd.DataFrame())
    empty_mock.assert_called_once()
