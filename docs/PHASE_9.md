# Phase 9 — Testing & Quality

## Architecture

Phase 9 adds a **layered pytest suite** aligned with clean architecture boundaries:

| Layer | Location | Focus |
|-------|----------|--------|
| Unit | `tests/unit/` | Pure logic: config, ML, preprocessing, performance, security |
| Integration | `tests/integration/` | Cross-module pipelines, DB persistence, training |
| API | `tests/api/` | FastAPI routes, auth, health, error contracts |
| Helpers | `tests/helpers/` | Factories, Streamlit mocking patterns |
| Fixtures | `tests/conftest.py` | Settings reset, in-memory SQLite, `TestClient` |

**Coverage policy:** `iad/` business logic is gated at **≥85%** (`pyproject.toml` → `fail_under = 85`). Pure Streamlit presentation modules are **omitted** from the denominator (see `[tool.coverage.run] omit` in `pyproject.toml`) because they require a browser runtime; they are validated via manual smoke (`streamlit run app.py`).

Omitted paths (presentation / optional deps):

- `iad/frontend/layouts/*`, `iad/frontend/auth/*`
- `iad/frontend/components/{layouts,uploaders,navbar,progress,charts,tables,model_cards,alerts,metric_cards}.py`
- `iad/frontend/performance/*`
- `iad/frontend/styles/theme.py`
- `iad/ml/automl/pycaret_adapter.py`

## Directory tree (new / updated)

```
tests/
├── api/
│   ├── test_health_and_errors.py
│   ├── test_auth_routes.py
│   └── test_auth_errors.py
├── helpers/
│   ├── factories.py
│   └── streamlit_mock.py
├── integration/
│   ├── test_persistence_training.py
│   └── test_log_prediction.py
├── unit/
│   ├── backend/
│   ├── frontend/
│   ├── ml/
│   ├── performance/
│   └── preprocessing/
├── conftest.py
└── fixtures/__init__.py
```

## Run tests

```bash
cd IAD
source .venv/bin/activate
pytest                    # full suite + coverage gate
pytest -m unit            # fast lane
pytest -m integration     # DB / pipeline
pytest --no-cov           # skip coverage gate (debug)
```

## Production fix included

- **Polars CSV `sep` → `separator`** mapping in `iad/performance/polars_io.py`
- **`.joblib`** added to `ALLOWED_FILE_EXTENSIONS` for model upload policy

## Migration / rollback

- **Migration:** no runtime change; CI should run `pytest` on every PR.
- **Rollback:** remove `[tool.pytest.ini_options] addopts` cov flags or lower `fail_under` temporarily.

## Next phase

Phase 10 — DevOps: Docker multi-stage, GitHub Actions, black/ruff/mypy/pre-commit.
