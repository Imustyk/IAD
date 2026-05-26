"""UI shell component tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_ui_render_helpers(monkeypatch) -> None:
    from iad.frontend.components import ui

    mock_render = MagicMock()
    monkeypatch.setattr(ui, "render_html_panel", mock_render)
    ui.render_hero(title="Test App", subtitle="Subtitle", badge="dev")
    ui.render_section_header("Section", "Details")
    ui.render_page_header("Page", "Caption")
    ui.render_empty_state("No data", "Upload a file", hint="CSV supported")
    assert mock_render.call_count == 4


@pytest.mark.unit
def test_theme_injects_bridge_css(monkeypatch) -> None:
    from iad.frontend.styles import theme

    mock_st = MagicMock()
    mock_st.session_state = {}
    monkeypatch.setattr(theme, "st", mock_st)
    theme.inject_css()
    combined = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "streamlit-bridge" in combined or "max-width: 1100px" in combined
