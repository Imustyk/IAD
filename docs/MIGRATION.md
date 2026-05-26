# Migration Plan

The platform is being upgraded from a coursework Streamlit app to a
production-grade enterprise ML SaaS. This document describes how the
migration is sequenced, which guarantees hold across phases and how to
roll back at any point.

## Guiding principles

1. **Strangler-fig.** New code lives in `iad/`. Legacy code in `src/` and
   `pages/` keeps working until its replacement is green.
2. **No big-bang rewrites.** Each phase is small, reviewed, tested and
   independently revertible.
3. **Backward-compatible session state.** The new `iad.state.session`
   re-exports the same string keys as `src.utils.SESSION_KEYS`, so
   existing pages keep reading and writing the same `st.session_state`.
4. **Tests are the gate.** Every phase ships with unit + integration
   tests covering both the new code and the legacy regression surface.
5. **Logging from day one.** Phase 1 installs structured + rotating
   logging so subsequent phases inherit observability for free.

## Phase 1 — Stabilise architecture (this iteration)

### What changed

* **Added** `iad/` package: `config`, `core`, `state` (real); `frontend`,
  `backend`, `ml`, `services` (placeholder packages with docstrings).
* **Added** `tests/` with unit tests for config / logging / exceptions /
  validation / paths / error handler, and an integration test that
  runs the legacy training pipeline end-to-end.
* **Added** `pyproject.toml` (modern packaging + ruff/black/mypy/pytest
  config), `.env.example`, `docs/ARCHITECTURE.md`, `docs/MIGRATION.md`,
  `docs/PHASE_1.md`.
* **Modified** `app.py`: two lines added at the top to install logging.
* **Modified** `requirements.txt`: added `pydantic`, `pydantic-settings`,
  `rich`, `pytest`, `pytest-cov`, `pytest-mock`.

### What did not change

* `pages/*.py` — untouched.
* `src/*.py` — untouched.
* `streamlit run app.py` — same command, same UX.
* Session state keys — identical.

### Rollback

```bash
git checkout pre-phase-1 -- app.py requirements.txt
rm -rf iad tests docs pyproject.toml .env.example
```

The application returns to the exact pre-Phase-1 state. Logs already
written remain in `logs/` and are harmless.

## Phase 2 — Data engineering layer ✅ delivered

* New: `iad/ml/preprocessing/` (Pandera schemas, GE adapter, quality scan,
  drift detection, 6 sklearn-compatible transformers, fluent pipeline
  builder + auto-factory, dataset profiler).
* New deps: `pandera>=0.18,<1.0`.
* Tests: 137 passing across unit (110) + sklearn-clone contract (13) +
  integration (3) + Phase-1 carry-over.
* Coverage: 80% across 1,651 statements.

### Rollback

```bash
git checkout pre-phase-2 -- requirements.txt pyproject.toml
rm -rf iad/ml/preprocessing tests/unit/preprocessing tests/integration/test_phase2_pipeline.py docs/PHASE_2.md
pip uninstall -y pandera
```

See `docs/PHASE_2.md` for full details.

## Phase 3 — ML platform ✅ delivered

* New: `iad/ml/{training,evaluation,tuning,explainability,tracking,automl}/`
  with a pluggable `ModelRegistry` (sklearn + XGBoost + LightGBM +
  CatBoost), `TrainingService` (leaderboard + cross-val + diagnostics +
  feature importance + model card), Optuna search across 12 model
  families, SHAP (Tree/Linear/Kernel auto-routing) + LIME, MLflow
  tracker (graceful fallback), FLAML AutoML adapter, PyCaret optional
  adapter, full reproducibility (`SeedManager` + `ModelCard` +
  `EnvironmentFingerprint` + dataset SHA-256).
* New deps: `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `lime`,
  `mlflow-skinny`, `flaml`.
* Tests: **175 passing** (44 net new in Phase 3) at **81% coverage**
  across 2,758 statements. Phase 1/2 regression tests still green.
* Migration target: Phase 4 will swap Streamlit's predictive page from
  `src.predictive` to `iad.ml.training.TrainingService`. `src.predictive`
  is retired once parity tests are green for two consecutive phases.

### Rollback

```bash
git checkout pre-phase-3 -- requirements.txt pyproject.toml
rm -rf iad/ml/{training,evaluation,tuning,explainability,tracking,automl} \
       tests/unit/ml tests/integration/test_phase3_pipeline.py docs/PHASE_3.md
pip uninstall -y xgboost lightgbm catboost optuna shap lime mlflow-skinny flaml
```

See `docs/PHASE_3.md` for full details.

## Phase 4 — Streamlit UX refactor ✅

* New: `iad/frontend/{styles,components,layouts,services}` — design system,
  reusable widgets, dashboard home, training bridge.
* `pages/*.py` remain at repo root (Streamlit discovery) but call `setup_page()`
  and shared components.
* Predictive page: optional **Enterprise ML engine** (`TrainingService`) with
  legacy `src.predictive` fallback via toggle / `IAD_UI_ENABLE_ENTERPRISE_ML`.
* Rollback: `git checkout HEAD -- app.py pages/` and remove new frontend dirs.
* Details: `docs/PHASE_4.md`.

## Phase 6 — Database & storage ✅

* New: `iad/backend/database`, `models`, `repositories`, `PersistenceService`.
* Alembic: `alembic upgrade head` (see `docs/PHASE_6.md`).
* Default SQLite at `data/iad.db`; PostgreSQL via `IAD_DATABASE_URL`.
* Opt-in: `IAD_DATABASE_ENABLED=true` (Streamlit unchanged by default).

## Phase 7 — Authentication & security ✅

* JWT auth API (`/auth/register`, `/login`, `/refresh`, `/me`)
* bcrypt passwords, RBAC, CSRF middleware, rate limits, secure headers
* Optional Streamlit login (`IAD_AUTH_ENABLED=true`)
* Run: `python scripts/run_api.py` — see `docs/PHASE_7.md`

## Phase 8 — Performance ✅

* New: `iad/performance/` (Polars, Dask, memory, lazy views, background jobs).
* Streamlit: cached previews, background training toggle, dtype optimisation on load.
* See `docs/PHASE_8.md`.

## Phase 5+ — ML API routes, testing, ops

See `docs/ARCHITECTURE.md` § "Future architecture" for details.

## Compatibility guarantees per phase

| Concern | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5+ |
|---|---|---|---|---|---|
| `streamlit run app.py` works | ✅ | ✅ | ✅ | ✅ | ✅ |
| `src/` public API stable | ✅ | ✅ | until parity | retired | retired |
| Session state keys stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tests required to merge | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rollback in one command | ✅ | ✅ | ✅ | ✅ | ✅ |
