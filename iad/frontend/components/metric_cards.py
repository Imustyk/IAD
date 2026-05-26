"""Reusable KPI / metric card components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st

DeltaDirection = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single KPI card."""

    label: str
    value: str
    delta: str | None = None
    delta_direction: DeltaDirection = "neutral"
    icon: str | None = None
    help_text: str | None = None


def _delta_class(direction: DeltaDirection) -> str:
    return direction


def render_metric_card(spec: MetricSpec) -> None:
    """Render a single styled metric card (HTML)."""
    icon_html = f'<span style="font-size:1.25rem;margin-right:0.5rem">{spec.icon}</span>' if spec.icon else ""
    delta_html = ""
    if spec.delta:
        delta_html = (
            f'<div class="iad-metric-delta {_delta_class(spec.delta_direction)}">'
            f"{spec.delta}</div>"
        )
    help_attr = f' title="{spec.help_text}"' if spec.help_text else ""
    st.markdown(
        f"""
        <div class="iad-metric-card iad-animate-in"{help_attr}>
          <div class="iad-metric-label">{icon_html}{spec.label}</div>
          <div class="iad-metric-value">{spec.value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(specs: list[MetricSpec], columns: int | None = None) -> None:
    """Render a responsive row of metric cards."""
    if not specs:
        return
    n = columns or len(specs)
    n = max(1, min(n, 4))
    cols = st.columns(n)
    for idx, spec in enumerate(specs):
        if idx < n:
            with cols[idx]:
                render_metric_card(spec)


def render_native_metrics(specs: list[MetricSpec]) -> None:
    """Fallback using native ``st.metric`` when HTML cards are undesirable."""
    if not specs:
        return
    cols = st.columns(min(len(specs), 4))
    for idx, spec in enumerate(specs):
        with cols[idx % len(cols)]:
            st.metric(
                label=spec.label,
                value=spec.value,
                delta=spec.delta,
                help=spec.help_text,
            )
