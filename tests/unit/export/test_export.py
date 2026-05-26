"""Phase 13 export & reporting tests."""
from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient

from iad.backend.api.app import create_app
from iad.config.settings import get_settings
from iad.core.exceptions import ExportError
from iad.export.charts import export_plotly_figure, kaleido_available
from iad.export.models import AnalyticsReport, ExportFormat, ReportSection
from iad.export.report_builder import build_report_from_dataframe, build_report_from_session
from iad.export.service import ExportService
from iad.state.session import (
    KEY_BUSINESS_CASE,
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_TARGET_COLUMN,
    KEY_TRAINING_REPORT,
)


@pytest.fixture
def exports_dir(tmp_path: Path) -> Path:
    return tmp_path / "exports"


@pytest.fixture
def sample_report() -> AnalyticsReport:
    return AnalyticsReport(
        title="Test Report",
        subtitle="Unit test",
        dataset_name="iris",
        dataset_shape=(150, 5),
        metrics={"accuracy": 0.92, "f1_macro": 0.91},
        sections=[
            ReportSection(
                title="Summary",
                body="Automated test section.",
                bullets=("Row 1 ok",),
                metrics={"rows": 150.0},
            )
        ],
        generated_at_iso="2026-05-26T00:00:00+00:00",
    )


def test_build_report_from_session_with_training() -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeTraining:
        best_model_name: str = "rf"
        metrics: dict[str, float] | None = None
        cv_metrics: dict[str, float] | None = None

        def leaderboard_frame(self):
            import pandas as pd

            return pd.DataFrame(
                [{"model_name": "rf", "accuracy": 0.9}, {"model_name": "xgb", "accuracy": 0.88}]
            )

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = build_report_from_session(
        {
            KEY_DATASET: df,
            KEY_DATASET_NAME: "demo",
            KEY_TARGET_COLUMN: "b",
            KEY_TRAINING_REPORT: FakeTraining(
                metrics={"accuracy": 0.9},
                cv_metrics={"accuracy": 0.88},
            ),
        }
    )
    assert report.model_name == "rf"
    assert report.metrics["accuracy"] == 0.9
    assert any(s.title == "Cross-validation" for s in report.sections)


def test_build_report_from_session() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = build_report_from_session(
        {
            KEY_DATASET: df,
            KEY_DATASET_NAME: "demo",
            KEY_TARGET_COLUMN: "b",
        }
    )
    assert report.dataset_name == "demo"
    assert report.dataset_shape == (2, 2)
    assert report.sections


def test_build_report_from_dataframe() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, None]})
    report = build_report_from_dataframe(df, dataset_name="toy")
    assert report.dataset_shape == (3, 1)
    assert report.sections[0].metrics["missing_cells_pct"] > 0


def test_export_pdf_and_docx(exports_dir: Path, sample_report: AnalyticsReport) -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    service = ExportService(exports_dir=exports_dir)
    pdf = service.export_report_pdf(sample_report)
    docx = service.export_report_docx(sample_report)
    assert pdf.path.exists() and pdf.size_bytes > 100
    assert docx.path.exists() and docx.size_bytes > 1000


def test_export_metrics_and_chart(exports_dir: Path, sample_report: AnalyticsReport) -> None:
    service = ExportService(exports_dir=exports_dir)
    csv_res = service.export_metrics(sample_report.metrics, fmt=ExportFormat.CSV)
    json_res = service.export_metrics(sample_report.metrics, fmt=ExportFormat.JSON)
    assert csv_res.path.suffix == ".csv"
    assert json_res.path.suffix == ".json"

    fig = px.scatter(x=[1, 2, 3], y=[1, 4, 9])
    html_res = service.export_chart(fig, fmt=ExportFormat.HTML)
    assert html_res.path.suffix == ".html"
    assert "plotly" in html_res.path.read_text(encoding="utf-8").lower()


def test_automated_export_bundle(exports_dir: Path, sample_report: AnalyticsReport) -> None:
    kaleido_available.cache_clear()
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    service = ExportService(exports_dir=exports_dir)
    bundle = service.generate_automated_report(sample_report, df=df, create_zip=True)
    assert bundle.manifest_path.exists()
    assert bundle.file_count >= 3
    if bundle.zip_path:
        assert bundle.zip_path.exists()


def test_build_report_includes_business_case() -> None:
    df = pd.DataFrame({"a": [1]})
    report = build_report_from_session(
        {
            KEY_DATASET: df,
            KEY_DATASET_NAME: "demo",
            KEY_BUSINESS_CASE: {
                "title": "Churn reduction",
                "problem": "High churn",
                "objective": "Predict churn",
                "kpis": "Retention",
                "stakeholders": "Product",
                "data_sources": "CRM",
            },
        }
    )
    assert any(s.title == "Churn reduction" for s in report.sections)


def test_export_bundle(exports_dir: Path, sample_report: AnalyticsReport) -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    service = ExportService(exports_dir=exports_dir)
    results = service.export_bundle(sample_report)
    assert len(results) >= 2


def test_png_export_requires_kaleido(exports_dir: Path) -> None:
    kaleido_available.cache_clear()
    fig = px.scatter(x=[1], y=[1])
    dest = exports_dir / "chart.png"
    if kaleido_available():
        result = export_plotly_figure(fig, dest, fmt=ExportFormat.PNG)
        assert result.path.exists()
    else:
        with pytest.raises(ExportError, match="kaleido"):
            export_plotly_figure(fig, dest, fmt=ExportFormat.PNG)


def test_kaleido_available_cached() -> None:
    kaleido_available.cache_clear()
    first = kaleido_available()
    second = kaleido_available()
    assert first is second


def test_kaleido_available_false_when_not_installed(monkeypatch) -> None:
    kaleido_available.cache_clear()
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "kaleido":
            raise ImportError("no kaleido")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert kaleido_available() is False


def test_kaleido_available_true_when_probe_succeeds(monkeypatch) -> None:
    kaleido_available.cache_clear()

    def _noop_write(self, path, **_kwargs) -> None:  # noqa: ANN001
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(go.Figure, "write_image", _noop_write)
    assert kaleido_available() is True


def test_export_unsupported_chart_format(exports_dir: Path) -> None:
    fig = px.scatter(x=[1], y=[1])
    with pytest.raises(ExportError, match="Unsupported"):
        export_plotly_figure(fig, exports_dir / "chart.pdf", fmt=ExportFormat.PDF)


def test_png_export_write_failure(monkeypatch, exports_dir: Path) -> None:
    kaleido_available.cache_clear()
    if not kaleido_available():
        pytest.skip("kaleido PNG export is not available in this environment")
    fig = px.scatter(x=[1], y=[1])

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("renderer down")

    monkeypatch.setattr(fig, "write_image", _boom)
    with pytest.raises(ExportError, match="PNG export failed"):
        export_plotly_figure(fig, exports_dir / "fail.png", fmt=ExportFormat.PNG)


def test_png_export_success_with_mocked_kaleido(monkeypatch, exports_dir: Path) -> None:
    kaleido_available.cache_clear()
    monkeypatch.setattr("iad.export.charts.kaleido_available", lambda: True)

    def _write_png(self, path, **_kwargs) -> None:  # noqa: ANN001
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(go.Figure, "write_image", _write_png)
    fig = px.scatter(x=[1, 2], y=[2, 4])
    result = export_plotly_figure(fig, exports_dir / "ok.png", fmt=ExportFormat.PNG)
    assert result.path.exists()
    assert result.content_type == "image/png"


def test_automated_export_skips_failed_png(
    monkeypatch, exports_dir: Path, sample_report: AnalyticsReport
) -> None:
    from iad.export import automated as auto_mod

    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    real_export = export_plotly_figure

    def _export(fig, destination, *, fmt=ExportFormat.HTML, **kwargs):
        if fmt == ExportFormat.PNG:
            raise ExportError("png failed", code="kaleido_export_failed")
        return real_export(fig, destination, fmt=fmt, **kwargs)

    monkeypatch.setattr(auto_mod, "kaleido_available", lambda: True)
    monkeypatch.setattr(auto_mod, "export_plotly_figure", _export)
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    service = ExportService(exports_dir=exports_dir)
    bundle = service.generate_automated_report(sample_report, df=df, create_zip=False)
    assert bundle.manifest_path.exists()
    assert not any(p.suffix == ".png" for p in bundle.run_dir.rglob("*"))


def test_api_export_docx(monkeypatch) -> None:
    pytest.importorskip("docx")
    monkeypatch.setenv("IAD_CSRF_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/exports/report/docx",
        json={"title": "API DOCX", "sections": [{"title": "S", "body": "body"}]},
    )
    assert response.status_code == 200
    assert response.json()["format"] == "docx"
    get_settings.cache_clear()


def test_api_export_automated(monkeypatch) -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    monkeypatch.setenv("IAD_CSRF_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/exports/report/automated",
        json={"title": "Auto", "metrics": {"accuracy": 0.9}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_count"] >= 2
    assert body["manifest_path"]
    get_settings.cache_clear()


def test_api_export_pdf(monkeypatch) -> None:
    pytest.importorskip("reportlab")
    monkeypatch.setenv("IAD_CSRF_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/exports/report/pdf",
        json={
            "title": "API Report",
            "metrics": {"accuracy": 0.9},
            "sections": [{"title": "T", "body": "B", "metrics": {"x": 1.0}}],
        },
    )
    assert response.status_code == 200
    assert response.json()["format"] == "pdf"
    get_settings.cache_clear()
