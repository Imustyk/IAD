"""Streamlit component render tests (mocked st calls)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.mark.unit
def test_metric_cards_render(monkeypatch) -> None:
    from iad.frontend.components import metric_cards

    monkeypatch.setattr(metric_cards, "render_html_panel", MagicMock())
    spec = metric_cards.MetricSpec(label="Accuracy", value="0.95", delta="+2%")
    metric_cards.render_metric_card(spec)
    metric_cards.render_metric_row([spec, spec], columns=2)


@pytest.mark.unit
def test_model_cards_render(monkeypatch) -> None:
    from iad.frontend.components import model_cards

    monkeypatch.setattr(model_cards, "st", MagicMock())
    entry = model_cards.LeaderboardEntry(
        rank=1,
        model_name="Random Forest",
        primary_metric=0.91,
        primary_metric_label="Accuracy",
        is_best=True,
        family="tree",
    )
    model_cards.render_model_card(entry)
    model_cards.render_leaderboard([entry])


@pytest.mark.unit
def test_alerts_and_guard(monkeypatch) -> None:
    from iad.frontend.components import alerts

    monkeypatch.setattr(alerts, "st", MagicMock())
    alerts.success("ok")
    alerts.warning("warn")
    alerts.info("info")
    alerts.error("err", exc=ValueError("x"))
    alerts.render_status_pill("Ready", "ok")
    assert alerts.guard_dataset_loaded(False) is False
    assert alerts.guard_dataset_loaded(True) is True


@pytest.mark.unit
def test_theme_helpers(monkeypatch) -> None:
    from iad.frontend.styles import theme

    mock_st = MagicMock()
    mock_st.session_state = {}
    monkeypatch.setattr(theme, "st", mock_st)
    assert theme.get_theme() == "light"
    theme.set_theme("dark")
    assert theme.get_theme() == "light"
    theme.inject_css()
    assert theme.get_theme() == "light"


@pytest.mark.unit
def test_leaderboard_table_and_chart(monkeypatch) -> None:
    from iad.frontend.components import model_cards

    monkeypatch.setattr(model_cards, "st", MagicMock())
    monkeypatch.setattr(model_cards, "st_dataframe", MagicMock())
    monkeypatch.setattr(model_cards, "render_plotly", MagicMock())
    df = pd.DataFrame({"model_name": ["a", "b"], "roc_auc": [0.9, 0.8]})
    model_cards.render_leaderboard_table(df, metric_col="roc_auc")
    model_cards.render_leaderboard_chart(df, metric_col="roc_auc")
