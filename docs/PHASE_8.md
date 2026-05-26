# Phase 8 — Performance

## Architecture

Performance is split into **engine-agnostic** (`iad/performance/`) and **Streamlit-specific** (`iad/frontend/performance/`) layers so the FastAPI backend can reuse Polars/Dask/memory utilities without importing Streamlit.

```
Upload / URL / Sample
        │
        ▼
  polars_io.read_*_fast  ──► prepare_for_session (dtype optimisation)
        │
        ▼
  st.session_state["dataset"]
        │
        ├── LazyDatasetView (preview sampling)
        ├── @st.cache_data (tables, charts, correlations)
        ├── Dask (value counts / describe when rows ≥ threshold)
        └── BackgroundJobRunner (optional training)
```

### ADR-020 — Opt-in heavy engines

| Setting | Default | Purpose |
|---------|---------|---------|
| `PERF_USE_POLARS` | true | Fast CSV/Parquet ingest |
| `PERF_USE_DASK` | true | Parallel stats above 100k rows |
| `PERF_AUTO_OPTIMIZE_DTYPES` | true | 30–70% RAM reduction |
| `PERF_BACKGROUND_TRAINING` | true | Thread-pool training |
| `PERF_LAZY_PREVIEW_ROWS` | 5000 | Cap preview/sample size |

### ADR-021 — Background jobs vs true async

Streamlit is synchronous; we use a **thread pool** (`BackgroundJobRunner`) so the UI reruns while sklearn trains. For multi-process scale-out, use the FastAPI `/train` endpoint (Phase 5) with a task queue (Celery/RQ) later.

## Directory tree

```
iad/performance/
├── fingerprints.py
├── memory.py
├── polars_io.py
├── dask_engine.py
├── lazy.py
├── jobs.py
└── loader.py

iad/frontend/performance/
├── streamlit_cache.py
└── background.py
```

## Integration points

| Location | Change |
|----------|--------|
| `context.store_dataset` | `prepare_for_session()` |
| Data Loading | Polars upload path + memory caption |
| Descriptive | Dask value counts when large |
| Predictive | Background training checkbox + job polling |
| `tables.py` | Shared `dataframe_fingerprint` |

## Testing

```bash
pytest tests/unit/performance/ -v
```

## Rollback

```bash
git checkout HEAD -- iad/performance iad/frontend/performance
# Set IAD_PERF_USE_POLARS=false IAD_PERF_USE_DASK=false
```

## Production notes

- Replace in-memory `RateLimiter` / `BackgroundJobRunner` with Redis + Celery when running multiple Streamlit replicas.
- Polars shines on CSV/Parquet; Excel still uses pandas/openpyxl.
- Dask adds overhead on small data — thresholds tune the crossover point.
