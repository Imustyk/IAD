"""Automated report generation — bundle PDF, DOCX, charts, metrics, and manifest."""
from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from iad.core.logging import get_logger
from iad.core.observability.performance import timed_block
from iad.export.charts import export_plotly_figure, kaleido_available
from iad.export.docx import build_docx_report
from iad.export.metrics_export import export_metrics_table
from iad.export.models import AnalyticsReport, AutomatedExportBundle, ExportFormat, ExportResult
from iad.export.pdf import build_pdf_report

logger = get_logger("iad.export.automated")


def _write_manifest(
    path: Path,
    *,
    report: AnalyticsReport,
    artifacts: list[ExportResult],
    run_dir: Path,
) -> None:
    payload = {
        "generated_at": report.generated_at_iso or datetime.now(tz=UTC).isoformat(),
        "title": report.title,
        "dataset_name": report.dataset_name,
        "model_name": report.model_name,
        "metrics": report.metrics,
        "run_dir": str(run_dir),
        "files": [
            {
                "format": item.format.value,
                "path": str(item.path.relative_to(run_dir)),
                "size_bytes": item.size_bytes,
                "content_type": item.content_type,
            }
            for item in artifacts
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(source_dir))
    return zip_path


def _default_charts(df: pd.DataFrame, columns: list[str] | None) -> list[tuple[str, go.Figure]]:
    numeric = [c for c in df.select_dtypes(include="number").columns.tolist() if c in (columns or df.columns)]
    if not numeric:
        return []
    chosen = numeric[:3]
    figures: list[tuple[str, go.Figure]] = []
    for col in chosen:
        fig = px.histogram(df, x=col, title=f"Distribution — {col}")
        figures.append((f"hist_{col}", fig))
    return figures


def generate_automated_report(
    report: AnalyticsReport,
    run_dir: Path,
    *,
    df: pd.DataFrame | None = None,
    chart_columns: list[str] | None = None,
    include_pdf: bool = True,
    include_docx: bool = True,
    include_metrics: bool = True,
    include_charts_html: bool = True,
    embed_charts_in_pdf: bool = True,
    create_zip: bool = True,
) -> AutomatedExportBundle:
    """Build a full export package under ``run_dir``.

    Steps:
        1. Optional chart exports (HTML; PNG when Kaleido is installed).
        2. PDF/DOCX with embedded PNG charts when available.
        3. Metrics CSV/JSON.
        4. ``manifest.json`` listing all artifacts.
        5. Optional ZIP of the run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ExportResult] = []
    chart_paths_for_pdf: list[Path] = []

    with timed_block("export_automated"):
        if df is not None and (include_charts_html or embed_charts_in_pdf):
            charts_dir = run_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            for name, fig in _default_charts(df, chart_columns):
                if include_charts_html:
                    html_dest = charts_dir / f"{name}.html"
                    artifacts.append(export_plotly_figure(fig, html_dest, fmt=ExportFormat.HTML))
                if embed_charts_in_pdf and kaleido_available():
                    png_dest = charts_dir / f"{name}.png"
                    artifacts.append(export_plotly_figure(fig, png_dest, fmt=ExportFormat.PNG))
                    chart_paths_for_pdf.append(png_dest)

        report_with_charts = AnalyticsReport(
            title=report.title,
            subtitle=report.subtitle,
            dataset_name=report.dataset_name,
            dataset_shape=report.dataset_shape,
            target_column=report.target_column,
            task_type=report.task_type,
            model_name=report.model_name,
            sections=list(report.sections),
            metrics=dict(report.metrics),
            chart_paths=list(report.chart_paths) + chart_paths_for_pdf,
            generated_at_iso=report.generated_at_iso,
        )

        if include_pdf:
            artifacts.append(build_pdf_report(report_with_charts, run_dir / "report.pdf"))
        if include_docx:
            artifacts.append(build_docx_report(report_with_charts, run_dir / "report.docx"))
        if include_metrics and report.metrics:
            artifacts.append(
                export_metrics_table(report.metrics, run_dir / "metrics.csv", fmt=ExportFormat.CSV)
            )
            artifacts.append(
                export_metrics_table(report.metrics, run_dir / "metrics.json", fmt=ExportFormat.JSON)
            )

        manifest_path = run_dir / "manifest.json"
        _write_manifest(manifest_path, report=report_with_charts, artifacts=artifacts, run_dir=run_dir)

        zip_path: Path | None = None
        if create_zip:
            zip_path = run_dir.parent / f"{run_dir.name}.zip"
            _zip_directory(run_dir, zip_path)

    logger.info(
        "automated export complete",
        extra={"ctx_run_dir": str(run_dir), "ctx_files": len(artifacts), "ctx_zip": str(zip_path)},
    )
    return AutomatedExportBundle(
        run_dir=run_dir,
        manifest_path=manifest_path,
        artifacts=artifacts,
        zip_path=zip_path,
    )
