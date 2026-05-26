"""KPI metric cards — embedded HTML for reliable rendering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st

from iad.frontend.components.html_render import esc, render_html_panel

DeltaDirection = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class MetricSpec:
    label: str
    value: str
    delta: str | None = None
    delta_direction: DeltaDirection = "neutral"
    icon: str | None = None
    help_text: str | None = None


def _delta_class(direction: DeltaDirection) -> str:
    return {"positive": "pos", "negative": "neg", "neutral": "neu"}.get(direction, "neu")


def render_metric_card(spec: MetricSpec) -> None:
    delta_html = ""
    if spec.delta:
        delta_html = f'<div class="iad-metric-delta {_delta_class(spec.delta_direction)}">{esc(spec.delta)}</div>'
    body = (
        f'<div class="iad-metric">'
        f'<div class="iad-metric-label">{esc(spec.label)}</div>'
        f'<div class="iad-metric-value">{esc(spec.value)}</div>'
        f"{delta_html}</div>"
    )
    render_html_panel(body, height=108)


def render_metric_row(specs: list[MetricSpec], columns: int | None = None) -> None:
    if not specs:
        return
    n = max(1, min(columns or len(specs), 4))
    cols = st.columns(n)
    for idx, spec in enumerate(specs):
        if idx < n:
            with cols[idx]:
                render_metric_card(spec)
