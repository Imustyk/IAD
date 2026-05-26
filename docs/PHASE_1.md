# Phase 1 — Stabilise architecture

## Goal

Lay the production foundations that every subsequent phase depends on:

* a real Python package layout,
* environment-driven configuration,
* structured + rotating logging,
* a typed exception hierarchy with a global error handler,
* a defensive validation layer for uploads, schemas and inference,
* a typed Streamlit session-state accessor,
* an automated pytest suite with a regression smoke test for the legacy
  pipeline.

## What was delivered

### New packages

```
iad/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py            # pydantic-settings Settings + get_settings()
├── core/
│   ├── __init__.py            # auto-installs logging on import
│   ├── logging.py             # rotating files (app/errors/training) + Rich
│   ├── exceptions.py          # IADError hierarchy
│   ├── error_handler.py       # page_guard, safe_action, handle_error
│   ├── paths.py               # safe_filename + dir helpers
│   └── validation.py          # uploads, dataframes, columns, inference
├── state/
│   ├── __init__.py
│   └── session.py             # typed st.session_state accessors
├── frontend/                  # placeholder for Phase 4
├── backend/                   # placeholder for Phase 5
├── ml/                        # placeholder for Phases 2-3, 11
└── services/                  # placeholder for Phase 5+
```

### Tests

```
tests/
├── conftest.py                # sys.path + settings fixture
├── unit/
│   ├── test_config.py         # 6 tests
│   ├── test_logging.py        # 5 tests
│   ├── test_exceptions.py     # 6 tests
│   ├── test_validation.py     # 14 tests
│   ├── test_paths.py          # 3 tests
│   └── test_error_handler.py  # 5 tests
└── integration/
    └── test_legacy_smoke.py   # 3 end-to-end tests on Iris
```

### Top-level

* `pyproject.toml` — packaging, pytest, coverage, ruff, black, mypy.
* `.env.example` — every supported env var documented.
* `requirements.txt` — extended with new deps.
* `docs/ARCHITECTURE.md` and `docs/MIGRATION.md`.

### Surgical edits

* `app.py` — gained two lines: `from iad.core import get_logger` and
  `from iad.config import get_settings`. Logging is installed on the
  first import; the body is unchanged.

## How it integrates

* Any module can do `from iad.core import get_logger, page_guard,
  validation` and immediately get fully-configured logging + a global
  error handler + the validation API.
* Any module can do `from iad.config import get_settings` to read
  env-driven configuration once.
* Streamlit hot-reload is safe: `configure_logging()` is idempotent
  and re-runs do not duplicate handlers.

## How to run it

```bash
# Install (incremental — adds the new deps to the existing venv):
pip install -r requirements.txt

# Run the platform unchanged:
streamlit run app.py

# Run the test suite:
pytest

# Run with coverage:
pytest --cov=iad --cov-report=term-missing
```

## Edge cases handled

* **Streamlit re-runs on every interaction.** `configure_logging()` is
  guarded by `_CONFIGURED` and removes prior handlers when called with
  `force=True`. No duplicate log lines.
* **Python `app.py` ↔ `app/` collision.** Avoided by using `iad/` as
  the canonical package name (ADR-001 in `ARCHITECTURE.md`).
* **`pydantic-settings` strict parsing.** All fields have safe
  defaults; `extra="ignore"` so an unknown env var does not crash boot.
* **Secret leakage in logs.** `Settings.safe_dict()` redacts
  `SECRET_KEY`, `DATABASE_URL`, `SENTRY_DSN`.
* **Legacy session-state contract.** New `iad.state.session` constants
  match the legacy `src.utils.SESSION_KEYS` values verbatim, so old
  pages keep working unmodified.
* **Streamlit not available in tests.** `iad.core.error_handler`
  imports Streamlit lazily; tests for the handler do not require a
  Streamlit runtime.

## Production considerations

* **Log shipping.** Set `IAD_LOG_JSON=true` in production for
  structured JSON logs that Loki / Datadog / ELK can index.
* **Log retention.** `RotatingFileHandler` caps each file at 10 MiB
  with 10 backups (configurable). For multi-day retention combine
  with logrotate or a sidecar shipper.
* **Secrets.** `IAD_SECRET_KEY` MUST be overridden via environment in
  production. The default is intentionally non-secret to surface the
  miss-config in audits.
* **Resource caps.** `IAD_MAX_UPLOAD_MB`, `IAD_MAX_ROWS`,
  `IAD_MAX_COLUMNS` are real safety rails — every upload path enforces
  them via `validate_uploaded_file` / `validate_dataframe`.
* **Health.** `streamlit /_stcore/health` already returns `ok`. Phase
  12 will add `/healthz` (FastAPI) with subsystem checks.

## What Phase 2 will build on top of this

* Pandera schemas for the seven sample datasets and any uploaded data.
* Custom sklearn-compatible transformers in `iad.ml.preprocessing`.
* Data drift detection between two snapshots of the same dataset.
* ydata-profiling integration with HTML/PDF export to `reports/`.
