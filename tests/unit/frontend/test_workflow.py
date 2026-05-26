"""Navigation and pipeline UI tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_nav_groups_structure() -> None:
    from iad.frontend.routes import NAV_GROUPS

    titles = [g.title for g in NAV_GROUPS]
    assert titles == ["Overview", "Data", "Analytics", "Reports"]
    data = next(g for g in NAV_GROUPS if g.title == "Data")
    assert len(data.items) == 1


@pytest.mark.unit
def test_pipeline_timeline_renders_html(monkeypatch) -> None:
    from iad.frontend.components import pipeline

    mock_render = MagicMock()
    monkeypatch.setattr(pipeline, "render_html_panel", mock_render)
    pipeline.render_pipeline_timeline()
    mock_render.assert_called_once()
    body = mock_render.call_args[0][0]
    assert "iad-pipeline-card" in body
    assert "iad-step-badge" in body
