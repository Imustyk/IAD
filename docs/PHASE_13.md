# Phase 13 — Export & Reporting

## 1. Architecture

Phase 13 delivers **auditable, downloadable artifacts** for thesis demos and production handoffs. Design principles:

| Principle | Implementation |
|-----------|----------------|
| **Service layer** | `ExportService` — single facade for UI and API |
| **Pure builders** | `pdf.py`, `docx.py`, `charts.py` — testable without Streamlit |
| **Session snapshot** | `report_builder.py` — maps `st.session_state` → `AnalyticsReport` |
| **Automation** | `automated.py` — one-click ZIP: PDF + DOCX + charts + metrics + `manifest.json` |
| **Observability** | `timed_block` + Prometheus `export_*` counters |

```
Streamlit page ──┐
                 ├──► ExportService ──► pdf / docx / charts / metrics
FastAPI /exports ┘              └──► automated.generate_automated_report()
```

**Why not export inside pages?** Keeps `pages/8_*.py` thin; API clients get identical output paths under `exports/`.

## 2. Directory tree

```
iad/export/
├── __init__.py
├── models.py              # AnalyticsReport, ExportResult, AutomatedExportBundle
├── report_builder.py      # Session → report (incl. business case, leaderboard)
├── pdf.py                 # ReportLab
├── docx.py                # python-docx
├── charts.py              # Plotly HTML/PNG (Kaleido optional)
├── metrics_export.py      # CSV/JSON
├── automated.py           # Manifest + ZIP bundle
└── service.py             # ExportService facade

pages/8_📄_Export_Reports.py
iad/backend/api/routes/exports.py
exports/                   # runtime artifacts (Docker volume)
docs/PHASE_13.md
tests/unit/export/test_export.py
```

## 3. Features delivered

| # | Requirement | Status |
|---|-------------|--------|
| 1 | PDF exports | `build_pdf_report` — tables, sections, embedded PNG charts |
| 2 | DOCX exports | `build_docx_report` |
| 3 | Downloadable charts | HTML always; PNG with `pip install -e ".[export]"` |
| 4 | Downloadable metrics | CSV + JSON |
| 5 | Automated report generation | `generate_automated_report` + ZIP + manifest |

## 4. Configuration

Artifacts use `IAD_EXPORTS_DIR` (default: `exports/`). No secrets in export paths.

## 5. API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/exports/report/pdf` | PDF from JSON payload |
| POST | `/exports/report/docx` | DOCX from JSON payload |
| POST | `/exports/report/automated` | Full package + manifest + ZIP |

## 6. Streamlit

Open **Export & Reports** (page 8):

- **Generate full automated package (ZIP)** — recommended one-click flow
- Individual PDF / DOCX / bundle buttons
- Per-column chart HTML/PNG export
- Metrics CSV/JSON when a model is trained

## 7. Testing

```bash
pytest tests/unit/export/test_export.py -q
pytest -m "not slow"   # full suite
```

## 8. Migration

1. `pip install -r requirements-prod.txt` (includes `reportlab`, `python-docx`)
2. Optional: `pip install -e ".[export]"` for Plotly PNG
3. Use page 8 or call API — no changes to existing analytics pages

## 9. Rollback

Remove `pages/8_*.py`, `iad/export/`, and `exports` router include in `iad/backend/api/app.py`. Core ML workflow unaffected.

## 10. Production considerations

- **Disk**: mount `exports/` with retention (ZIPs grow with chart count)
- **Concurrency**: each run uses a unique `report_<uuid>/` directory
- **Kaleido**: optional; PDF chart embeds skipped if not installed (HTML charts still exported)
- **PII**: reports include session dataset profile — scrub before external sharing
