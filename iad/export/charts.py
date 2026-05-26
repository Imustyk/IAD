"""Plotly figure export — HTML (always) and PNG (when Kaleido is available)."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from iad.core.exceptions import ExportError
from iad.core.logging import get_logger
from iad.export.models import ExportFormat, ExportResult

logger = get_logger("iad.export.charts")


def kaleido_available() -> bool:
    try:
        import kaleido  # noqa: F401

        return True
    except ImportError:
        return False


def export_plotly_figure(
    fig: go.Figure,
    destination: Path,
    *,
    fmt: ExportFormat = ExportFormat.HTML,
    width: int = 1200,
    height: int = 700,
) -> ExportResult:
    """Write a Plotly figure to disk in the requested format."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt == ExportFormat.HTML:
        fig.write_html(str(destination), include_plotlyjs="cdn", full_html=True)
        return ExportResult(
            format=fmt,
            path=destination,
            size_bytes=destination.stat().st_size,
            content_type="text/html",
        )

    if fmt == ExportFormat.PNG:
        if not kaleido_available():
            raise ExportError(
                "PNG export requires kaleido. Install with: pip install kaleido",
                code="kaleido_missing",
            )
        fig.write_image(str(destination), width=width, height=height, scale=2)
        return ExportResult(
            format=fmt,
            path=destination,
            size_bytes=destination.stat().st_size,
            content_type="image/png",
        )

    raise ExportError(f"Unsupported chart format: {fmt}", code="unsupported_chart_format")
