"""Report export API — programmatic PDF/DOCX generation."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from iad.export.models import AnalyticsReport, ReportSection
from iad.export.service import ExportService

router = APIRouter(prefix="/exports", tags=["exports"])


class ReportSectionIn(BaseModel):
    title: str
    body: str = ""
    bullets: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class ReportExportIn(BaseModel):
    title: str = "IAD Analytics Report"
    subtitle: str = ""
    dataset_name: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    sections: list[ReportSectionIn] = Field(default_factory=list)


class ExportPathOut(BaseModel):
    format: str
    path: str
    size_bytes: int


class AutomatedExportOut(BaseModel):
    run_dir: str
    manifest_path: str
    zip_path: str | None
    file_count: int
    files: list[ExportPathOut]


@router.post("/report/pdf", response_model=ExportPathOut)
def export_pdf(payload: ReportExportIn) -> ExportPathOut:
    report = AnalyticsReport(
        title=payload.title,
        subtitle=payload.subtitle,
        dataset_name=payload.dataset_name,
        metrics=payload.metrics,
        sections=[
            ReportSection(
                s.title,
                s.body,
                tuple(s.bullets),
                dict(s.metrics),
            )
            for s in payload.sections
        ],
    )
    result = ExportService().export_report_pdf(report)
    return ExportPathOut(format=result.format.value, path=str(result.path), size_bytes=result.size_bytes)


@router.post("/report/docx", response_model=ExportPathOut)
def export_docx(payload: ReportExportIn) -> ExportPathOut:
    report = AnalyticsReport(
        title=payload.title,
        subtitle=payload.subtitle,
        dataset_name=payload.dataset_name,
        metrics=payload.metrics,
        sections=[
            ReportSection(
                s.title,
                s.body,
                tuple(s.bullets),
                dict(s.metrics),
            )
            for s in payload.sections
        ],
    )
    result = ExportService().export_report_docx(report)
    return ExportPathOut(format=result.format.value, path=str(result.path), size_bytes=result.size_bytes)


@router.post("/report/automated", response_model=AutomatedExportOut)
def export_automated(payload: ReportExportIn) -> AutomatedExportOut:
    """Generate PDF, DOCX, metrics, charts (when possible), manifest, and ZIP."""
    report = AnalyticsReport(
        title=payload.title,
        subtitle=payload.subtitle,
        dataset_name=payload.dataset_name,
        metrics=payload.metrics,
        sections=[
            ReportSection(
                s.title,
                s.body,
                tuple(s.bullets),
                dict(s.metrics),
            )
            for s in payload.sections
        ],
    )
    bundle = ExportService().generate_automated_report(report, create_zip=True)
    return AutomatedExportOut(
        run_dir=str(bundle.run_dir),
        manifest_path=str(bundle.manifest_path),
        zip_path=str(bundle.zip_path) if bundle.zip_path else None,
        file_count=bundle.file_count,
        files=[
            ExportPathOut(format=a.format.value, path=str(a.path), size_bytes=a.size_bytes)
            for a in bundle.artifacts
        ],
    )
