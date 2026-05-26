# Phase 5 — FastAPI ML Backend

## Architecture

Phase 5 exposes the same ML capabilities as Streamlit through a **REST API**:

| Layer | Responsibility |
|-------|----------------|
| **Routes** (`iad/backend/api/routes/ml.py`) | HTTP, auth, async offload |
| **Schemas** (`iad/backend/schemas/ml.py`) | Pydantic validation |
| **Services** | Training, inference, upload, catalog |
| **ML core** (`iad/ml/training`) | Unchanged training logic |
| **Persistence** | Optional DB + always-on `.joblib` artifacts |

**Why `asyncio.to_thread`?** Training and inference are CPU-bound; threads keep the event loop responsive without duplicating sklearn code.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health`, `/healthz`, `/live` | No | Probes (Phase 12) |
| GET | `/metrics` | No | Prometheus (not ML metrics) |
| POST | `/auth/*` | Mixed | JWT auth |
| POST | `/upload` | Bearer | Dataset file upload |
| POST | `/train` | Bearer | Train leaderboard |
| POST | `/predict` | Bearer | Batch inference |
| GET | `/models` | Bearer | List models |
| GET | `/experiments` | Bearer | List experiments |
| GET | `/ml/metrics` | Bearer | Stored ML metrics |
| POST | `/exports/*` | Bearer/CSRF-exempt | Reports (Phase 13) |

## Quick start

```bash
# Register + token
curl -s -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"ml@example.com","password":"Password123","full_name":"ML User"}'

export TOKEN="<access_token>"

# Upload
curl -s -X POST "http://localhost:8000/upload?name=demo" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/samples/telco_churn.csv"

# Train (inline JSON — see OpenAPI for full schema)
curl -s -X POST http://localhost:8000/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target_column":"species","records":[...]}'

# Predict
curl -s -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"artifact_path":"models/....joblib","records":[{"sepal_length":5.1}]}'
```

OpenAPI: http://localhost:8000/docs

## Tests

```bash
pytest tests/api/test_ml_routes.py -q
```

## Rollback

Remove `ml.router` from `iad/backend/api/app.py` and delete new service/schema files. Streamlit-only workflow unaffected.
