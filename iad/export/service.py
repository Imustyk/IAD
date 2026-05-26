"""Export service — orchestrates PDF, DOCX, charts, and metrics downloads."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.core.observability.performance import timed_block
from iad.core.observability.prometheus import observe_ml_operation
from iad.export.automated import generate_automated_report
from iad.export.charts import export_plotly_figure
from iad.export.docx import build_docx_report
from iad.export.metrics_export import export_metrics_table
from iad.export.models import AnalyticsReport, AutomatedExportBundle, ExportFormat, ExportResult
from iad.export.pdf import build_pdf_report

logger = get_logger("iad.export.service")


class ExportService:
    """Facade for report generation and artifact export."""

    def __init__(self, exports_dir: Path | None = None) -> None:
        settings = get_settings()
        self._exports_dir = exports_dir or settings.EXPORTS_DIR
        self._exports_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, prefix: str = "report") -> Path:
        run_id = uuid.uuid4().hex[:12]
        path = self._exports_dir / f"{prefix}_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_report_pdf(self, report: AnalyticsReport, *, filename: str = "report.pdf") -> ExportResult:
        with timed_block("export_pdf"):
            out = self._run_dir("pdf") / filename
            result = build_pdf_report(report, out)
            observe_ml_operation(operation="export_pdf", outcome="success")
            return result

    def export_report_docx(self, report: AnalyticsReport, *, filename: str = "report.docx") -> ExportResult:
        with timed_block("export_docx"):
            out = self._run_dir("docx") / filename
            result = build_docx_report(report, out)
            observe_ml_operation(operation="export_docx", outcome="success")
            return result

    def export_chart(
        self,
        fig: go.Figure,
        *,
        fmt: ExportFormat = ExportFormat.HTML,
        filename: str | None = None,
    ) -> ExportResult:
        ext = "html" if fmt == ExportFormat.HTML else "png"
        name = filename or f"chart.{ext}"
        with timed_block(f"export_chart_{ext}"):
            out = self._run_dir("chart") / name
            result = export_plotly_figure(fig, out, fmt=fmt)
            observe_ml_operation(operation="export_chart", outcome="success")
            return result

    def export_metrics(
        self,
        metrics: dict[str, float],
        *,
        fmt: ExportFormat = ExportFormat.CSV,
        filename: str | None = None,
    ) -> ExportResult:
        ext = "json" if fmt == ExportFormat.JSON else "csv"
        name = filename or f"metrics.{ext}"
        with timed_block("export_metrics"):
            out = self._run_dir("metrics") / name
            result = export_metrics_table(metrics, out, fmt=fmt)
            observe_ml_operation(operation="export_metrics", outcome="success")
            return result

    def export_bundle(
        self,
        report: AnalyticsReport,
        *,
        include_pdf: bool = True,
        include_docx: bool = True,
        include_metrics: bool = True,
    ) -> list[ExportResult]:
        """Generate a full export bundle (PDF + DOCX + metrics CSV)."""
        results: list[ExportResult] = []
        run_dir = self._run_dir("bundle")
        if include_pdf:
            results.append(build_pdf_report(report, run_dir / "report.pdf"))
        if include_docx:
            results.append(build_docx_report(report, run_dir / "report.docx"))
        if include_metrics and report.metrics:
            results.append(
                export_metrics_table(report.metrics, run_dir / "metrics.csv", fmt=ExportFormat.CSV)
            )
        logger.info("export bundle complete", extra={"ctx_files": len(results), "ctx_dir": str(run_dir)})
        return results

    def generate_automated_report(
        self,
        report: AnalyticsReport,
        *,
        df: pd.DataFrame | None = None,
        chart_columns: list[str] | None = None,
        create_zip: bool = True,
    ) -> AutomatedExportBundle:
        """One-click package: reports, charts, metrics, manifest, and optional ZIP."""
        run_dir = self._run_dir("automated")
        bundle = generate_automated_report(
            report,
            run_dir,
            df=df,
            chart_columns=chart_columns,
            create_zip=create_zip,
        )
        observe_ml_operation(operation="export_automated", outcome="success")
        return bundle
