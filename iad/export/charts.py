"""Plotly figure export — HTML (always) and PNG (when Kaleido is available)."""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go

from iad.core.exceptions import ExportError
from iad.core.logging import get_logger
from iad.export.models import ExportFormat, ExportResult

logger = get_logger("iad.export.charts")


@lru_cache(maxsize=1)
def kaleido_available() -> bool:
    """True when kaleido is installed and can render at least one PNG."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False

    try:
        probe = go.Figure(data=[go.Scatter(x=[0], y=[0])])
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            probe.write_image(tmp.name, width=8, height=8)
        return True
    except Exception as exc:  # pragma: no cover — environment-specific
        logger.debug("kaleido installed but PNG export unavailable: %s", exc)
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
                "PNG export requires a working kaleido install (pip install kaleido)",
                code="kaleido_missing",
            )
        try:
            fig.write_image(str(destination), width=width, height=height, scale=2)
        except Exception as exc:
            raise ExportError(
                f"PNG export failed: {exc}",
                code="kaleido_export_failed",
            ) from exc
        return ExportResult(
            format=fmt,
            path=destination,
            size_bytes=destination.stat().st_size,
            content_type="image/png",
        )

    raise ExportError(f"Unsupported chart format: {fmt}", code="unsupported_chart_format")
