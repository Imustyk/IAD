"""Export & reporting — PDF, DOCX, charts, and metrics downloads."""
from __future__ import annotations

import json

import streamlit as st

from iad.core.error_handler import handle_error
from iad.export import ExportService
from iad.export.models import ExportFormat
from iad.export.report_builder import build_report_from_session
from iad.frontend.layouts.page import section, setup_page
from iad.frontend.streamlit_compat import button, download_button
from src.utils import require_dataset

setup_page(
    "Export & Reports",
    caption="Phase 13 — download PDF/DOCX reports, charts, and metric tables.",
)

df = require_dataset()
if df is None:
    st.stop()

service = ExportService()
report = build_report_from_session(dict(st.session_state))

section("Report preview")
st.markdown(f"**{report.title}** — {report.subtitle}")
for line in report.summary_lines():
    st.write(line)
if report.metrics:
    st.json(report.metrics)

section("Automated report export")

if button("Generate full automated package (ZIP)", type="primary"):
    try:
        bundle = service.generate_automated_report(report, df=df, create_zip=True)
        st.success(
            f"Package ready — {bundle.file_count} artifact(s) in `{bundle.run_dir.name}`"
        )
        if bundle.zip_path and bundle.zip_path.exists():
            with open(bundle.zip_path, "rb") as zf:
                download_button(
                    "Download ZIP bundle",
                    data=zf.read(),
                    file_name=bundle.zip_path.name,
                    mime="application/zip",
                )
        with open(bundle.manifest_path, encoding="utf-8") as mf:
            st.json(json.load(mf))
        for item in bundle.artifacts[:6]:
            with open(item.path, "rb") as fh:
                st.download_button(
                    f"Download {item.path.name}",
                    data=fh.read(),
                    file_name=item.path.name,
                    mime=item.content_type,
                    key=f"auto_{item.path.name}",
                )
        if bundle.file_count > 6:
            st.caption(f"+ {bundle.file_count - 6} more files in `{bundle.run_dir}`")
    except Exception as exc:
        handle_error(exc)

col_pdf, col_docx, col_bundle = st.columns(3)

with col_pdf:
    if button("Generate PDF", type="primary"):
        try:
            result = service.export_report_pdf(report)
            st.success(f"PDF ready ({result.size_bytes:,} bytes)")
            with open(result.path, "rb") as fh:
                download_button(
                    "Download PDF",
                    data=fh.read(),
                    file_name=result.path.name,
                    mime=result.content_type,
                )
        except Exception as exc:
            handle_error(exc)

with col_docx:
    if button("Generate DOCX"):
        try:
            result = service.export_report_docx(report)
            st.success(f"DOCX ready ({result.size_bytes:,} bytes)")
            with open(result.path, "rb") as fh:
                download_button(
                    "Download DOCX",
                    data=fh.read(),
                    file_name=result.path.name,
                    mime=result.content_type,
                )
        except Exception as exc:
            handle_error(exc)

with col_bundle:
    if button("Full bundle"):
        try:
            results = service.export_bundle(report)
            st.success(f"Bundle created — {len(results)} file(s)")
            for item in results:
                with open(item.path, "rb") as fh:
                    st.download_button(
                        f"Download {item.path.name}",
                        data=fh.read(),
                        file_name=item.path.name,
                        mime=item.content_type,
                        key=f"dl_{item.path.name}",
                    )
        except Exception as exc:
            handle_error(exc)

section("Chart & metrics export")
num_cols = [c for c in df.columns if str(df[c].dtype) != "object"]
if num_cols:
    import plotly.express as px

    chart_col = st.selectbox("Column for quick chart", num_cols)
    fig = px.histogram(df, x=chart_col, title=f"Distribution — {chart_col}")
    c1, c2 = st.columns(2)
    if c1.button("Export chart (HTML)"):
        try:
            res = service.export_chart(fig, fmt=ExportFormat.HTML)
            with open(res.path, "rb") as fh:
                st.download_button("Download HTML chart", fh.read(), res.path.name, res.content_type)
        except Exception as exc:
            handle_error(exc)
    if c2.button("Export chart (PNG)"):
        try:
            res = service.export_chart(fig, fmt=ExportFormat.PNG)
            with open(res.path, "rb") as fh:
                st.download_button("Download PNG chart", fh.read(), res.path.name, res.content_type)
        except Exception as exc:
            handle_error(exc)

if report.metrics:
    m1, m2 = st.columns(2)
    if m1.button("Export metrics (CSV)"):
        try:
            res = service.export_metrics(report.metrics, fmt=ExportFormat.CSV)
            with open(res.path, "rb") as fh:
                st.download_button("Download metrics CSV", fh.read(), res.path.name, res.content_type)
        except Exception as exc:
            handle_error(exc)
    if m2.button("Export metrics (JSON)"):
        try:
            res = service.export_metrics(report.metrics, fmt=ExportFormat.JSON)
            with open(res.path, "rb") as fh:
                st.download_button("Download metrics JSON", fh.read(), res.path.name, res.content_type)
        except Exception as exc:
            handle_error(exc)

st.caption("Exports are written under `exports/` and can be mounted as a Docker volume.")
