# IAD — Architecture

## Vision

A production-grade enterprise ML SaaS platform built on Streamlit (UI),
FastAPI (API), and a clean Python package (`iad`) that owns business logic,
ML pipelines and persistence.

## Layered model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Delivery channels                            │
│  ┌────────────────────┐               ┌──────────────────────────┐  │
│  │ Streamlit frontend │               │  FastAPI backend (Phase 5)│  │
│  │ (pages, components)│               │  (REST API + OpenAPI)    │  │
│  └─────────┬──────────┘               └────────────┬─────────────┘  │
│            │                                       │                │
└────────────┼───────────────────────────────────────┼────────────────┘
             │                                       │
             ▼                                       ▼
   ┌──────────────────────────────────────────────────────────┐
   │              iad.services — use-case layer               │
   │  (orchestration shared by both delivery channels)        │
   └─────────┬─────────────────────┬──────────────────┬───────┘
             │                     │                  │
             ▼                     ▼                  ▼
   ┌──────────────┐    ┌────────────────────┐    ┌─────────────────┐
   │   iad.ml     │    │ iad.backend        │    │ iad.core        │
   │ (training,   │    │ (repos, db, sec)   │    │ (logging,       │
   │  eval, SHAP, │    │ (Phase 6+)         │    │  exceptions,    │
   │  Optuna, ...)│    │                    │    │  validation,    │
   │ (Phase 2-3)  │    │                    │    │  paths)         │
   └──────────────┘    └────────────────────┘    └─────────────────┘
                                                       ▲
                                                       │
                                              ┌────────┴────────┐
                                              │  iad.config     │
                                              │ (pydantic-      │
                                              │  settings)      │
                                              └─────────────────┘
```

## Package map

| Package | Status | Responsibility |
|---|---|---|
| `iad.config` | ✅ Phase 1 | Settings, environment loading, validation |
| `iad.core` | ✅ Phase 1 | Logging, exceptions, error handler, validation, paths |
| `iad.state` | ✅ Phase 1 | Typed Streamlit session-state accessors |
| `iad.frontend` | ⏳ Phase 4 | Streamlit components, layouts, styles, pages |
| `iad.backend` | ✅ Phase 6 (partial) | SQLAlchemy models, repositories, Alembic; API routes Phase 5 |
| `iad.ml` | ⏳ Phases 2-3, 11 | Preprocessing, training, evaluation, explainability, tuning, AutoML, NLP, forecasting, clustering, anomaly, recommendation |
| `iad.services` | ⏳ Phase 5+ | Cross-channel use-case orchestration |

## Architectural Decision Records

### ADR-001 — Use `iad/` as the canonical Python package

**Context.** The original spec uses `app/`, `backend/`, `ml/` as top-level
directories. The Streamlit entry file `app.py` is invoked directly by
`streamlit run app.py`. Having both `app.py` and `app/` at the same path
creates fragile import semantics (Python's package-over-module precedence
plus the script-as-`__main__` rule).

**Decision.** Use a single named package `iad/` for all production code.
`iad.frontend`, `iad.backend`, `iad.ml`, `iad.services` map 1:1 to the
spec's `app/`, `backend/`, `ml/`. `app.py` stays as the Streamlit entry.

**Consequence.** No naming collision; idiomatic Python; tests and CI work
without `PYTHONPATH` gymnastics.

### ADR-002 — Strangler-fig migration over rewrite

**Context.** The legacy `src/` and `pages/` modules already work end-to-end
and are exercised by the user.

**Decision.** Add new code in `iad/` next to the legacy code. Migrate one
concern per phase. Retire legacy modules only after parity tests are green.

**Consequence.** Each phase ships independently and is independently
revertible. The application is never broken between phases.

### ADR-003 — pydantic-settings over `os.environ` and YAML

**Context.** Settings need to be typed, validated, environment-aware and
serialisable for logging.

**Decision.** Use `pydantic-settings>=2.2`. One `Settings` class, one
`get_settings()` cached singleton, one `.env.example` template.

**Consequence.** All configuration is documented in code (the `Settings`
class) and validated at boot. No runtime "string lookup" on env vars.

## Runtime data flow (Phase 1)

```
.env  ─┐
       ├──▶  Settings (pydantic-settings)  ──▶  iad.core.logging  ──┐
ENV   ─┘                                                            │
                                                                    ▼
            iad.core.exceptions  ◀────  iad.core.error_handler  ──▶ Streamlit UI
                                                ▲
                                                │
                            iad.core.validation ┘
```

## Future architecture (Phases 2-13)

See `docs/ROADMAP.md` (created at the start of each phase) for the per-phase
delta. The end state matches the user's target tree under `iad/` and adds:

* `docker/` + `docker-compose.yml` (Phase 10)
* `.github/workflows/ci.yml` (Phase 10)
* `iad/backend/database/migrations/` (Alembic, Phase 6)
* `iad/ml/tracking/mlflow.py` (Phase 3)
* Prometheus + Sentry in `iad.core.observability` (Phase 12) — `/metrics`, `/live`, Grafana profile
* PDF/DOCX/chart export in `iad.export` (Phase 13)
