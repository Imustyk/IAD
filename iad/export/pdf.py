"""PDF report generation using ReportLab."""
from __future__ import annotations

from pathlib import Path

from iad.core.exceptions import ExportError
from iad.core.logging import get_logger
from iad.export.models import AnalyticsReport, ExportFormat, ExportResult

logger = get_logger("iad.export.pdf")


def build_pdf_report(report: AnalyticsReport, destination: Path) -> ExportResult:
    """Render an analytics report as a PDF document."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ExportError(
            "PDF export requires reportlab. Install with: pip install reportlab",
            code="reportlab_missing",
        ) from exc

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "IADTitle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1e3a5f"),
    )
    section_style = ParagraphStyle(
        "IADSection",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#2563eb"),
    )
    body_style = styles["BodyText"]

    story: list[object] = []
    story.append(Paragraph(report.title, title_style))
    if report.subtitle:
        story.append(Paragraph(report.subtitle, body_style))
    if report.generated_at_iso:
        story.append(Paragraph(f"<i>Generated: {report.generated_at_iso}</i>", body_style))
    story.append(Spacer(1, 0.4 * cm))

    for line in report.summary_lines():
        story.append(Paragraph(line, body_style))

    if report.metrics:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Model metrics", section_style))
        rows = [["Metric", "Value"]] + [[k, f"{v:.6g}" if isinstance(v, float) else str(v)] for k, v in report.metrics.items()]
        table = Table(rows, colWidths=[8 * cm, 6 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        story.append(table)

    for section in report.sections:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(section.title, section_style))
        story.append(Paragraph(section.body.replace("\n", "<br/>"), body_style))
        for bullet in section.bullets:
            story.append(Paragraph(f"• {bullet}", body_style))
        if section.metrics:
            metric_rows = [["Metric", "Value"]] + [
                [k, f"{v:.6g}" if isinstance(v, float) else str(v)] for k, v in section.metrics.items()
            ]
            mtable = Table(metric_rows, colWidths=[8 * cm, 6 * cm])
            mtable.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
            story.append(Spacer(1, 0.2 * cm))
            story.append(mtable)

    for chart_path in report.chart_paths:
        if chart_path.exists() and chart_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(str(chart_path), width=14 * cm, height=8 * cm))

    try:
        doc.build(story)
    except Exception as exc:
        logger.exception("PDF build failed")
        raise ExportError(f"PDF generation failed: {exc}", code="pdf_build_failed") from exc

    logger.info("PDF exported", extra={"ctx_path": str(destination)})
    return ExportResult(
        format=ExportFormat.PDF,
        path=destination,
        size_bytes=destination.stat().st_size,
        content_type="application/pdf",
    )
