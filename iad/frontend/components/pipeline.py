"""Analytics pipeline timeline — fully rendered HTML panel."""
from __future__ import annotations

from dataclasses import dataclass

from iad.frontend.components.html_render import esc, render_html_panel


@dataclass(frozen=True)
class PipelineStep:
    number: int
    title: str
    description: str


DEFAULT_PIPELINE: tuple[PipelineStep, ...] = (
    PipelineStep(1, "Data loading", "Import CSV, Excel, JSON, or sample datasets"),
    PipelineStep(2, "Descriptive analytics", "Summaries, distributions, and quality checks"),
    PipelineStep(3, "Diagnostic analytics", "Correlations, tests, and outlier review"),
    PipelineStep(4, "Predictive analytics", "Train, tune, and compare ML models"),
    PipelineStep(5, "Apply model", "Score single rows or batch files"),
    PipelineStep(6, "Prescriptive analytics", "What-if scenarios and recommendations"),
    PipelineStep(7, "Advanced analytics", "NLP, forecasting, clustering, anomalies"),
    PipelineStep(8, "Export and reports", "PDF, DOCX, charts, and report bundles"),
)


def render_pipeline_timeline(steps: tuple[PipelineStep, ...] | list[PipelineStep] | None = None) -> None:
    """Vertical/grid pipeline cards with embedded CSS (always renders correctly)."""
    items = list(steps or DEFAULT_PIPELINE)
    cards = []
    for step in items:
        cards.append(
            f'<article class="iad-pipeline-card">'
            f'<span class="iad-step-badge">{step.number}</span>'
            f'<div class="iad-pipeline-body">'
            f'<div class="iad-pipeline-title">{esc(step.title)}</div>'
            f'<div class="iad-pipeline-desc">{esc(step.description)}</div>'
            f"</div></article>"
        )
    body = f'<div class="iad-wrap"><div class="iad-pipeline">{"".join(cards)}</div></div>'
    render_html_panel(body)
