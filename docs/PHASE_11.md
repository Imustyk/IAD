# Phase 11 — Advanced Analytics

## Architecture

Phase 11 adds **five analytical domains** under `iad/ml/`, each with:

- **Reports** — typed dataclasses for UI / API consumption
- **Algorithms** — pure functions (testable without Streamlit)
- **Services** — thin facades for pages and future FastAPI routes

```
iad/ml/
├── nlp/                 # sentiment, embeddings, LDA topics
├── forecasting/         # decomposition, ARIMA, Prophet
├── clustering/          # KMeans, DBSCAN, PCA/UMAP
├── anomaly/             # IsolationForest, OneClassSVM
└── recommendation/      # user CF, item cosine similarity
```

**Why services + functions?** Pages call services; unit tests call functions. FastAPI Phase 5+ can inject the same services without duplicating logic.

**Optional dependencies** (graceful degradation):

| Extra | Install | Enables |
|-------|---------|---------|
| (core) | `vaderSentiment` | Sentiment |
| `pip install -e ".[nlp]"` | sentence-transformers | Dense embeddings |
| `pip install -e ".[forecasting]"` | prophet | Prophet forecasts |
| `pip install -e ".[clustering]"` | umap-learn | UMAP plots |
| `pip install -e ".[analytics]"` | all of the above | Full Phase 11 |

## Streamlit integration

New page: `pages/7_🧠_Advanced_Analytics.py` — tabs for each domain. Existing pages unchanged.

## Run tests

```bash
pytest tests/unit/ml/test_phase11_*.py -q
pip install vaderSentiment  # if not already installed
```

## Migration / rollback

- **Enable:** `pip install -r requirements-prod.txt` (includes vaderSentiment)
- **Disable:** remove page file `pages/7_*.py` — no impact on core workflow
- **Rollback:** delete `iad/ml/{nlp,forecasting,clustering,anomaly,recommendation}` packages

## Production notes

- Prophet and sentence-transformers are **heavy** — keep them optional in API workers
- ARIMA fit can fail on short or constant series — UI shows user-safe errors via `handle_error`
- Recommendations need **long-format** interactions (user, item, rating)
