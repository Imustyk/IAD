"""Typed models for report generation and export results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExportFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PNG = "png"


@dataclass(frozen=True)
class ReportSection:
    """One logical block in an analytics report."""

    title: str
    body: str
    bullets: tuple[str, ...] = ()
    metrics: dict[str, float | str] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Structured report payload built from session / training artifacts."""

    title: str
    subtitle: str = ""
    dataset_name: str | None = None
    dataset_shape: tuple[int, int] | None = None
    target_column: str | None = None
    task_type: str | None = None
    model_name: str | None = None
    sections: list[ReportSection] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    chart_paths: list[Path] = field(default_factory=list)
    generated_at_iso: str = ""

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.dataset_name:
            shape = ""
            if self.dataset_shape:
                shape = f" ({self.dataset_shape[0]:,} rows × {self.dataset_shape[1]} cols)"
            lines.append(f"Dataset: {self.dataset_name}{shape}")
        if self.target_column:
            lines.append(f"Target: {self.target_column} ({self.task_type or 'unknown task'})")
        if self.model_name:
            lines.append(f"Champion model: {self.model_name}")
        return lines


@dataclass(frozen=True)
class ExportResult:
    """Result of a single export operation."""

    format: ExportFormat
    path: Path
    size_bytes: int
    content_type: str


@dataclass
class AutomatedExportBundle:
    """One-click export package (see :mod:`iad.export.automated`)."""

    run_dir: Path
    manifest_path: Path
    artifacts: list[ExportResult] = field(default_factory=list)
    zip_path: Path | None = None

    @property
    def file_count(self) -> int:
        return len(self.artifacts)
