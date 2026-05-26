"""KPI metric cards — single responsive grid panel per row."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def _metric_card_html(spec: MetricSpec) -> str:
    delta_html = ""
    if spec.delta:
        delta_html = (
            f'<div class="iad-metric-delta {_delta_class(spec.delta_direction)}">'
            f"{esc(spec.delta)}</div>"
        )
    return (
        f'<div class="iad-metric">'
        f'<div class="iad-metric-label">{esc(spec.label)}</div>'
        f'<div class="iad-metric-value">{esc(spec.value)}</div>'
        f"{delta_html}</div>"
    )


def render_metric_card(spec: MetricSpec) -> None:
    render_html_panel(_metric_card_html(spec))


def render_metric_row(specs: list[MetricSpec], columns: int | None = None) -> None:
    if not specs:
        return
    n = max(1, min(columns or len(specs), 4))
    cards = "".join(_metric_card_html(spec) for spec in specs[:n])
    body = f'<div class="iad-wrap"><div class="iad-metric-grid iad-metric-grid-{n}">{cards}</div></div>'
    render_html_panel(body)
