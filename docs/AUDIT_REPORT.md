# Production Audit Report — IAD Platform

**Date:** 2026-05-26  
**Scope:** Full-stack ML SaaS (Streamlit + FastAPI + SQLAlchemy + sklearn)  
**Test gate:** `pytest -m "not slow"` — **85%+ coverage**, all tests passing after remediation

---

## Executive summary

The codebase is **structurally sound** (layered services, typed config, tests, Docker/CI). This audit found **3 critical** and several **high/medium** issues; **critical items are fixed** in-tree.

| Severity | Found | Fixed in this pass |
|----------|-------|-------------------|
| Critical | 3 | 3 |
| High | 8 | 5 |
| Medium | 12 | 4 |
| Low | 15+ | documented |

---

## Critical issues (fixed)

### C1 — Shared preprocessor refit across leaderboard models

| Field | Detail |
|-------|--------|
| **Path** | `iad/ml/training/service.py` — `_train_single` |
| **Root cause** | Same `preprocessor` instance passed into every candidate `Pipeline`; later `fit()` calls refit shared state, invalidating earlier champions. |
| **Production risk** | Wrong model on disk, incorrect metrics, silent leaderboard corruption. |
| **Fix** | `clone(preprocessor)` per candidate before `Pipeline(...)`. |
| **Test** | `tests/unit/ml/test_preprocessor_clone.py` |

### C2 — Path traversal on `POST /predict` `artifact_path`

| Field | Detail |
|-------|--------|
| **Path** | `iad/backend/services/inference_service.py` |
| **Root cause** | Arbitrary filesystem paths accepted (`../../../etc/passwd`). |
| **Production risk** | Arbitrary file read via joblib load (RCE if malicious pickle). |
| **Fix** | `resolve_trusted_artifact_path()` in `iad/core/paths.py`; only `models/`, `data/uploads/`, `exports/`. |
| **Test** | `tests/unit/core/test_trusted_paths.py` |

### C3 — Apply Model page: unsafe `joblib.load` + wrong bundle schema

| Field | Detail |
|-------|--------|
| **Path** | `pages/5_🎯_Apply_Model.py` |
| **Root cause** | Raw `joblib.load(upload)`; expected keys `best_model_name` not in IAD bundle format. |
| **Production risk** | Arbitrary code execution; broken UX after load. |
| **Fix** | Write to `data/uploads/model_bundles/`, use `load_bundle()` + `handle_error`. |

---

## High issues

### H1 — Pickle deserialization (inherent)

| Field | Detail |
|-------|--------|
| **Paths** | `iad/ml/training/persistence.py`, `pages/5_*`, `streamlit_cache` |
| **Risk** | Malicious `.joblib` can execute code on load. |
| **Mitigation applied** | Trusted paths, extension check, size cap, structural validation (`pipeline` + `model_card` keys). |
| **Recommended** | Sign bundles; onnx/skops export for untrusted sources. |

### H2 — IDOR on ML catalog (fixed)

| Field | Detail |
|-------|--------|
| **Path** | `iad/backend/services/ml_catalog_service.py` |
| **Fix** | Verify `experiment.user_id` / `model.user_id` before listing metrics/models. |

### H3 — Upload DB registration ignored API session (fixed)

| Field | Detail |
|-------|--------|
| **Path** | `iad/backend/services/dataset_upload_service.py` |
| **Fix** | Use `session is not None` instead of `DATABASE_ENABLED and session`. |

### H4 — API task inference for string targets (fixed earlier)

| Field | Detail |
|-------|--------|
| **Path** | `iad/backend/services/training_api_service.py` |
| **Issue** | CSV `species` as `str` dtype inferred as regression. |

### H5 — `get_db` commits on read-only GET

| Field | Detail |
|-------|--------|
| **Risk** | Unnecessary commits; rare edge cases with implicit flushes. |
| **Recommendation** | Read-only dependency without auto-commit for GET routes. |

### H6 — CSRF exemptions broad

| Field | Detail |
|-------|--------|
| **Path** | `iad/backend/middleware/csrf.py` |
| **Risk** | Cookie-auth clients on `/train` without CSRF if misconfigured. |
| **Mitigation** | Bearer tokens skip CSRF; document cookie clients must send `X-CSRF-Token`. |

### H7 — Rate limit skips ML endpoints

| Field | Detail |
|-------|--------|
| **Recommendation** | Add `/train`, `/predict`, `/upload` to rate limiter with stricter limits. |

### H8 — No async training queue

| Field | Detail |
|-------|--------|
| **Risk** | Long trains block threadpool workers. |
| **Recommendation** | Celery/RQ + job status API (future). |

---

## ML / data engineering

| Check | Status |
|-------|--------|
| Train/test split before preprocess | OK |
| Target encoder OOF | OK (`SmoothedTargetEncoder`) |
| Preprocessor leakage across models | **Fixed** (clone) |
| Stratified split for classification | OK |
| Reproducibility seeds | OK (`SeedManager`) |
| Inference feature alignment | OK (`validate_inference_payload`) |

---

## Streamlit

| Issue | Severity | Notes |
|-------|----------|-------|
| Full dataframe in `session_state` | Medium | Expected; use `LazyDatasetView` for large data |
| `cached_model_bundle` raw joblib | Medium | Only used with trusted bytes id |
| Missing `page_guard` on some pages | Low | `handle_error` used on Apply Model |
| Rerun recomputation | Low | `@st.cache_data` on charts |

---

## FastAPI

| Check | Status |
|-------|--------|
| Blocking ML in threadpool | OK |
| IADError handler | OK |
| OpenAPI models | OK |
| Lifespan DB init | OK |

---

## Security checklist

| Item | Status |
|------|--------|
| JWT auth | OK |
| bcrypt passwords | OK |
| Upload validation | OK |
| Path traversal (artifacts) | **Fixed** |
| CORS configurable | OK |
| Security headers middleware | OK |
| Secrets in settings | OK (env-driven) |

---

## DevOps / testing

| Item | Status |
|------|--------|
| Docker multi-stage | OK |
| GitHub Actions CI | OK |
| Coverage ≥85% | OK |
| Pre-commit | OK |

---

## Remediation backlog (prioritized)

1. Stricter rate limits on `/train`, `/predict`, `/upload`
2. Read-only DB sessions for GET catalog routes
3. Signed model artifacts or alternative serialization
4. Background training jobs + status polling
5. Row-level dataset ownership checks on `dataset_id` in train
6. Streamlit `page_guard` on all pages
7. Replace raw `joblib.load` in `streamlit_cache.cached_model_bundle`

---

## Verification

```bash
pytest -m "not slow"
ruff check iad tests
mypy iad
streamlit run app.py
uvicorn iad.backend.api.app:app --reload
```
