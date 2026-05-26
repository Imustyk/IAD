# Phase 12 — Observability

## Architecture

Phase 12 adds **production observability** without changing Streamlit workflows:

| Component | Location | Purpose |
|-----------|----------|---------|
| Prometheus metrics | `iad/core/observability/prometheus.py` | Counters/histograms for HTTP + ML ops |
| HTTP middleware | `iad/backend/middleware/prometheus.py` | Automatic request latency |
| Scrape endpoint | `GET /metrics` | Prometheus text exposition |
| Sentry | `iad/core/observability/sentry.py` | Optional error + trace reporting |
| Health probes | `/health`, `/live`, `/healthz` | Liveness vs readiness |
| Performance hooks | `timed_block`, `observe_duration` | ML/export timing |

**Why a dedicated registry?** Tests stay isolated; cardinality is controlled via `normalize_path()`.

## Docker observability stack

```bash
docker compose --profile observability up -d
```

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin by default) |
| API metrics | http://localhost:8000/metrics |

Pre-provisioned dashboard: **IAD API Overview** (`docker/grafana/dashboards/iad-api-overview.json`).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `IAD_METRICS_ENABLED` | `true` | Expose `/metrics` |
| `IAD_SENTRY_DSN` | — | Enable Sentry when set |
| `IAD_SENTRY_ENABLED` | `true` | Master switch for Sentry |
| `IAD_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Performance trace sampling |

Streamlit unhandled errors are forwarded to Sentry via `iad.core.error_handler`.

## Run tests

```bash
pytest tests/unit/core/test_observability.py -q
```

## Rollback

Disable metrics: `IAD_METRICS_ENABLED=false`. Remove observability profile from Compose. Delete `iad/core/observability/` only if reverting the entire phase.
