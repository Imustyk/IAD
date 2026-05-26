# Intelligent Analytics and Data Science (IAD)

**Version 0.2.0**

IAD is a production-oriented data science and machine learning platform that combines a Streamlit analytics workspace, a FastAPI backend, and a structured Python package (`iad`) for training, evaluation, explainability, and reporting. It implements the full analytics lifecycle—from data ingestion through descriptive, diagnostic, predictive, and prescriptive analysis—while supporting optional PostgreSQL persistence, REST APIs, observability, and containerized deployment.

The application was built for master's-level coursework in **Data Analysis Tools** and has evolved into an enterprise-style SaaS layout with background training, model bundles, exports, and advanced analytics modules.

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Analytics workflow](#analytics-workflow)
4. [Architecture](#architecture)
5. [Prerequisites](#prerequisites)
6. [Quick start (local development)](#quick-start-local-development)
7. [Docker deployment](#docker-deployment)
8. [Configuration](#configuration)
9. [Application pages](#application-pages)
10. [Project structure](#project-structure)
11. [Technology stack](#technology-stack)
12. [REST API](#rest-api)
13. [Testing and quality gates](#testing-and-quality-gates)
14. [Troubleshooting](#troubleshooting)
15. [Documentation](#documentation)
16. [License](#license)

---

## Overview

IAD delivers two primary interfaces:

| Channel | Role |
|---------|------|
| **Streamlit UI** | Multi-page dashboard for analysts: load data, explore, train models, score, export reports |
| **FastAPI backend** | REST API with OpenAPI docs: health checks, dataset upload, training, prediction, auth (when enabled) |

Business logic lives in the `iad` package. Legacy helpers under `src/` remain for backward compatibility during migration; new features are implemented in `iad/` first.

Session state in Streamlit holds the active dataset, trained pipeline, and model metadata. When the database is enabled, datasets, users, and model records can persist in PostgreSQL.

---

## Features

### Data ingestion

- Upload CSV, TSV, TXT, Excel, JSON, or Parquet (up to configurable size limits)
- Load from public URLs
- Seven curated sample datasets: Iris, Wine, Breast Cancer, Diabetes, California Housing, Titanic-style classification, Telco churn
- Optional Polars-accelerated loading for large files
- Data hygiene tools: datetime parsing, duplicate removal, sparse column dropping

### Descriptive analytics

- Summary statistics (mean, median, skew, kurtosis, variance, range)
- Missing-value analysis and distribution plots (histogram, violin, box)
- Categorical frequency charts and time-series resampling

### Diagnostic analytics

- Pearson, Spearman, and Kendall correlation matrices
- Driver ranking, scatter matrices, IQR outlier detection
- Welch t-test, one-way ANOVA, Chi-square independence tests

### Predictive analytics

- Automatic classification vs regression detection
- Leak-safe preprocessing: imputation, scaling, one-hot encoding
- Model benchmarking: Logistic Regression, Linear/Ridge, Random Forest, Gradient Boosting, Decision Tree, KNN, plus XGBoost, LightGBM, CatBoost when installed
- Cross-validation, confusion matrix, residual plots, feature importances
- SHAP and LIME explainability (enterprise path)
- Optuna hyperparameter search and FLAML AutoML adapters
- Background training with progress polling (non-blocking UI)
- Model persistence via `cloudpickle` bundles (`.joblib`)

### Apply model

- Single-row manual scoring forms
- Batch CSV scoring with downloadable predictions
- Import/export of saved model bundles

### Prescriptive analytics

- Plain-language recommendations from model and data context
- One-feature sweeps and two-feature what-if heatmaps

### Advanced analytics

- Time-series decomposition and forecasting (ARIMA; Prophet when installed)
- Sentiment analysis (VADER lexicon)
- Recommendation engines (collaborative filtering, similarity)
- Clustering and anomaly detection modules (optional dependencies)

### Export and reporting

- PDF and DOCX reports with metrics tables and charts
- Automated export manifests (JSON, CSV, PDF)

### Platform capabilities

- Typed configuration via `pydantic-settings` (all `IAD_*` environment variables)
- Structured logging (console, rotating `app.log`, `errors.log`, `training.log`)
- Optional JWT authentication and RBAC (API and Streamlit gate)
- Prometheus metrics and Grafana dashboards (Docker observability profile)
- Alembic database migrations
- CI-ready: pytest with 85% coverage gate, Ruff, Black, mypy, pre-commit

---

## Analytics workflow

The UI follows the standard analytics progression:

| Phase | Question | Where in the app |
|-------|----------|------------------|
| Business case | What problem are we solving? | Home dashboard |
| Data sources | What data do we have? | Home, Data Loading |
| Descriptive | What happened? | Descriptive Analysis |
| Diagnostic | Why did it happen? | Diagnostic Analysis |
| Predictive | What will happen? | Predictive Modeling |
| Prescriptive | What should we do? | Prescriptive Analysis |
| Operationalize | Score new data | Apply Model |
| Advanced | NLP, forecasting, recommendations | Advanced Analytics |
| Deliver | Reports and exports | Export and Reports |

---

## Architecture

```
Delivery
  Streamlit (pages/, app.py)          FastAPI (iad/backend/api/)
           \                               /
            \                             /
             v                           v
         iad.frontend                 iad.backend
         (components, layouts)        (routes, security, repos)
                       \           /
                        v         v
                    iad.ml (training, preprocessing, explainability)
                    iad.services (orchestration)
                    iad.core (logging, exceptions, validation, paths)
                    iad.config (Settings)
                              |
                    PostgreSQL (optional)
```

Key design choices are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), including the use of the `iad/` package name (to avoid collision with `app.py`), strangler-fig migration from `src/`, and centralized settings.

---

## Prerequisites

- **Python 3.11+**
- **pip** and a virtual environment (recommended)
- **Docker** and **Docker Compose** (optional, for full stack)
- **Git**

For local API + database features: PostgreSQL client libraries are installed via `psycopg[binary]` when using the production requirements.

---

## Quick start (local development)

### 1. Clone and install

```bash
git clone <repository-url>
cd IAD
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # editable install with dev tools
# or: pip install -r requirements-dev.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed (see Configuration section)
```

### 3. Run the Streamlit UI

Use the helper script (recommended—it sets stable server flags for uploads):

```bash
python scripts/run_streamlit.py
```

Open **http://127.0.0.1:8501** in your browser.

Alternatively:

```bash
streamlit run app.py
```

### 4. Run the API (optional)

In a second terminal:

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

API documentation: **http://127.0.0.1:8000/docs**

The sidebar shows backend connectivity when the API is reachable.

### Important: one Streamlit instance per port

Do not run Docker Streamlit and local Streamlit on the same host port. Docker defaults to **8502**; local development uses **8501**. If file upload fails with HTTP 400, hard-refresh the browser (Cmd+Shift+R) and ensure only one server is bound to the port you are using.

---

## Docker deployment

### Full stack

```bash
cp .env.example .env
docker compose up --build
```

| Service | Default URL | Notes |
|---------|-------------|-------|
| Streamlit UI | http://localhost:8502 | Host port 8502 maps to container 8501 |
| FastAPI | http://localhost:8000/docs | Runs migrations on startup when configured |
| PostgreSQL | localhost:5433 | Host port 5433 maps to container 5432 |

### Optional profiles

```bash
# Prometheus + Grafana
docker compose --profile observability up -d

# Nginx reverse proxy on port 80
docker compose --profile proxy up -d
```

### Build targets

The multi-stage `docker/Dockerfile` supports:

```bash
docker build -f docker/Dockerfile --target streamlit -t iad-streamlit .
docker build -f docker/Dockerfile --target api -t iad-api .
```

Volumes persist `data/`, `models/`, `logs/`, and `exports/` across restarts.

---

## Configuration

All settings use the `IAD_` prefix. See [`.env.example`](.env.example) and [`iad/config/settings.py`](iad/config/settings.py) for the full list.

| Variable | Purpose |
|----------|---------|
| `IAD_ENVIRONMENT` | `development`, `staging`, `production`, or `test` |
| `IAD_DEBUG` | Verbose behavior and error details |
| `IAD_MAX_UPLOAD_MB` | Maximum upload size (aligned with Streamlit `maxUploadSize`) |
| `IAD_DATABASE_URL` | PostgreSQL connection string (optional) |
| `IAD_DATABASE_ENABLED` | Enable DB-backed features in the API |
| `IAD_AUTH_ENABLED` | Streamlit login gate (requires database) |
| `IAD_SECRET_KEY` | JWT and signing secret (override in production) |
| `IAD_PERF_USE_POLARS` | Polars-accelerated I/O |
| `IAD_PERF_BACKGROUND_TRAINING` | Non-blocking training jobs in the UI |
| `IAD_API_BASE_URL` | Backend URL shown in the sidebar (e.g. `http://127.0.0.1:8000`) |
| `IAD_STREAMLIT_PORT` | Docker host port for Streamlit (default `8502`) |

Streamlit-specific options are in [`.streamlit/config.toml`](.streamlit/config.toml) (light theme, upload size, file watcher disabled for stable uploads).

---

## Application pages

| Page | File | Description |
|------|------|-------------|
| Dashboard | `app.py` | Home, business case, pipeline overview |
| Data Loading | `pages/1_Data_Loading.py` | Upload, URL, samples, preview |
| Descriptive Analysis | `pages/2_Descriptive_Analysis.py` | Distributions, missing values, summaries |
| Diagnostic Analysis | `pages/3_Diagnostic_Analysis.py` | Correlations, tests, outliers |
| Predictive Modeling | `pages/4_Predictive_Modeling.py` | Train, compare, explain models |
| Apply Model | `pages/5_Apply_Model.py` | Score rows or batches |
| Prescriptive Analysis | `pages/6_Prescriptive_Analysis.py` | Recommendations and what-if |
| Advanced Analytics | `pages/7_Advanced_Analytics.py` | Forecasting, NLP, recommendations |
| Export and Reports | `pages/8_Export_Reports.py` | PDF/DOCX and automated exports |

Navigation is defined in [`iad/frontend/routes.py`](iad/frontend/routes.py) and rendered in the sidebar via [`iad/frontend/components/navbar.py`](iad/frontend/components/navbar.py).

---

## Project structure

```
.
├── app.py                          # Streamlit entry (home dashboard)
├── pages/                          # Streamlit multipage scripts
├── src/                            # Legacy loaders and analytics (migration source)
├── iad/                            # Canonical production package
│   ├── config/                     # pydantic-settings
│   ├── core/                       # logging, exceptions, validation, observability
│   ├── state/                      # Streamlit session accessors
│   ├── frontend/                   # UI components, layouts, styles, services
│   ├── backend/                    # FastAPI app, models, migrations, security
│   ├── ml/                         # preprocessing, training, tuning, explainability
│   ├── services/                   # cross-channel orchestration
│   ├── performance/                # Polars, Dask, background jobs
│   └── export/                     # report builders
├── tests/                          # unit, integration, API tests
├── scripts/                        # run_streamlit.py, run_api.py, db_init.py, ci.sh
├── docker/                         # Dockerfile, nginx, Grafana, Prometheus
├── docker-compose.yml
├── alembic.ini
├── .streamlit/config.toml
├── requirements.txt                # runtime pins
├── requirements-prod.txt           # production / Docker install
├── requirements-dev.txt            # developers
├── pyproject.toml                  # package metadata, pytest, ruff, mypy
└── docs/                           # phase guides and architecture
```

---

## Technology stack

| Layer | Technologies |
|-------|----------------|
| UI | Streamlit 1.36+, custom CSS design tokens, Plotly |
| API | FastAPI, Uvicorn, Pydantic v2 |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, FLAML, SHAP, LIME |
| Data | pandas, numpy, Polars, Dask (optional paths) |
| Persistence | SQLAlchemy 2, Alembic, PostgreSQL / SQLite |
| Security | bcrypt, python-jose (JWT), upload policy validation |
| Observability | prometheus-client, Sentry SDK (optional) |
| Export | ReportLab, python-docx |
| Quality | pytest, pytest-cov, Ruff, Black, mypy, pre-commit |

Optional extras (install via pip extras in `pyproject.toml`): `nlp`, `forecasting`, `clustering`, `analytics`, `export`.

---

## REST API

When the API container or `scripts/run_api.py` is running:

- **Health:** `GET /health`
- **Readiness:** `GET /healthz` (database and dependencies)
- **OpenAPI:** `/docs` and `/redoc`
- **ML routes:** upload datasets, train, predict, list models (see [docs/PHASE_5.md](docs/PHASE_5.md))

Authentication and CSRF apply when enabled; see [docs/PHASE_7.md](docs/PHASE_7.md).

---

## Testing and quality gates

```bash
# Full CI script (lint, typecheck, tests)
./scripts/ci.sh

# Tests only (exclude slow marker)
pytest -m "not slow"

# With coverage (85% minimum on iad package)
pytest
```

Install pre-commit hooks:

```bash
pre-commit install
```

Markers: `unit`, `integration`, `slow`. Coverage omits pure-Streamlit presentation modules listed in `pyproject.toml`.

---

## Troubleshooting

### File upload returns HTTP 400

Streamlit returns 400 when the browser session ID does not match a live server session. Common causes:

1. **Two Streamlit processes on the same port** (e.g. Docker and local both on 8501). Use Docker on **8502** or local on **8501**, not both.
2. **Stale tab after server restart.** Hard-refresh (Cmd+Shift+R) or open a new tab.
3. **Server restart during upload.** Wait for the app to finish loading, then upload again.

Configuration already disables the file watcher and XSRF protection for upload reliability (see `.streamlit/config.toml` and Docker environment variables).

### Sidebar shows "API offline"

Start the API: `python scripts/run_api.py --host 127.0.0.1 --port 8000`. Set `IAD_API_BASE_URL` if the API is not on the default host. Health checks are cached for 30 seconds in the sidebar.

### Database connection errors in Docker

Ensure Postgres is healthy (`docker compose ps`). Default host port is **5433** to avoid clashing with a local Postgres on 5432. Migrations run when `IAD_RUN_MIGRATIONS=true` on the API service.

### Port conflicts

| Service | Local default | Docker host default |
|---------|---------------|---------------------|
| Streamlit | 8501 | 8502 |
| API | 8000 | 8000 |
| PostgreSQL | — | 5433 |

---

## Documentation

| Document | Topic |
|----------|-------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, ADRs, package map |
| [docs/MIGRATION.md](docs/MIGRATION.md) | Legacy `src/` to `iad/` migration |
| [docs/PHASE_1.md](docs/PHASE_1.md) through [docs/PHASE_13.md](docs/PHASE_13.md) | Incremental delivery notes |
| [docs/PHASE_5.md](docs/PHASE_5.md) | REST API |
| [docs/PHASE_10.md](docs/PHASE_10.md) | CI, Docker, pre-commit |
| [docs/PHASE_11.md](docs/PHASE_11.md) | Advanced analytics |
| [docs/PHASE_12.md](docs/PHASE_12.md) | Observability |
| [docs/PHASE_13.md](docs/PHASE_13.md) | Export and reporting |

---

## Suggested walkthrough

1. **Dashboard** — review the business case and pipeline status.
2. **Data Loading** — load the Telco churn sample (or upload a CSV).
3. **Descriptive Analysis** — inspect distributions and missing values.
4. **Diagnostic Analysis** — explore correlations and run statistical tests.
5. **Predictive Modeling** — train models with target `churn`; review leaderboard and confusion matrix; download the bundle.
6. **Apply Model** — score a single record, then a batch file.
7. **Prescriptive Analysis** — read recommendations and run what-if sweeps.
8. **Export and Reports** — generate a PDF or DOCX summary.

---

## Extending the platform

- **New sample dataset:** add an entry to `SAMPLE_DATASETS` in `src/data_loader.py`.
- **New model family:** register in the training service / legacy `src/predictive.py` classifiers or regressors map.
- **New page:** add `pages/N_Name.py` and register in `iad/frontend/routes.py`.
- **New API route:** add under `iad/backend/api/routes/` and wire in `iad/backend/api/app.py`.

---

## License

Educational use as part of the **Data Analysis Tools** master's course. See project maintainers for production or redistribution terms.
