# Phase 10 — DevOps & Deployment

## Architecture

Phase 10 adds **repeatable build → test → ship** pipelines without changing application behaviour.

| Concern | Tooling | Rationale |
|---------|---------|-----------|
| Container images | Multi-stage `docker/Dockerfile` | Smaller runtime, non-root user, split UI/API |
| Local stack | `docker-compose.yml` | Postgres + API + Streamlit (+ optional nginx) |
| CI | GitHub Actions `ci.yml` | Block merges on lint, types, tests, image build |
| Registry | `docker-publish.yml` | Push `ghcr.io/<org>/<repo>/iad-{streamlit,api}` on release |
| Format / lint | Ruff + Black | Fast, standard Python toolchain |
| Types | Mypy on `iad/` | Catch contract drift before runtime |
| Git hooks | pre-commit | Same gates locally as CI |

**Why two images (streamlit + api)?** Different process models, scaling, and health checks. A single “fat” container would couple UI reloads with API deploys and complicate horizontal scaling.

**Why `PYTHONPATH=/app` instead of `pip install -e .` in Docker?** The image already installs wheels from `requirements-prod.txt`; copying `iad/` + `src/` keeps builds deterministic and avoids editable-install edge cases in slim images.

## Directory tree (new / updated)

```
IAD/
├── .dockerignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── requirements-prod.txt
├── requirements-dev.txt
├── docker/
│   ├── Dockerfile              # targets: streamlit | api
│   ├── entrypoint.sh           # optional alembic
│   ├── docker-compose.postgres.yml  # legacy single-DB compose
│   └── nginx/nginx.conf        # profile: proxy
├── .github/workflows/
│   ├── ci.yml
│   └── docker-publish.yml
└── scripts/
    ├── lint.sh
    ├── format.sh
    └── ci.sh
```

## Quick start

### Local (unchanged)

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

### Docker Compose (full stack)

```bash
cp .env.example .env
# Set IAD_SECRET_KEY to a strong value
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| nginx (optional) | `docker compose --profile proxy up` → http://localhost |

### Pre-commit

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

### Local CI mirror

```bash
./scripts/ci.sh
```

## CI pipeline

1. **lint** — `ruff check`, `ruff format --check`, `black --check`
2. **typecheck** — `mypy iad`
3. **test** — `pytest -m "not slow"` with ≥85% coverage on `iad/`
4. **docker** — build both image targets (no push)

## Production considerations

- Override `IAD_SECRET_KEY`, disable `IAD_DEBUG`, set `IAD_ENVIRONMENT=production`.
- Use PostgreSQL (`IAD_DATABASE_URL`) with `IAD_RUN_MIGRATIONS=true` on API startup or run Alembic in init job.
- Mount volumes for `data/`, `models/`, `logs/` (Compose defines named volumes).
- Pin image digests in production; publish via `docker-publish` on semver tags.
- Run `pip-audit` / Dependabot (recommended follow-up).

## Migration path

1. Developers adopt `requirements-dev.txt` and pre-commit.
2. CI runs on PRs; fix any first-time lint findings with `./scripts/format.sh`.
3. Staging deploy via Compose or GHCR images.
4. Production: same images + managed Postgres + secrets manager.

## Rollback

- **CI**: revert workflow commit; branch protection stops bad merges.
- **Deploy**: roll Kubernetes/Compose to previous image tag.
- **Local**: `streamlit run app.py` still works without Docker.

## Edge cases

- **`.env` missing**: Compose fails fast — copy `.env.example` first.
- **CatBoost / LightGBM in slim image**: builder stage installs compile deps; runtime uses `libgomp1`.
- **Streamlit health**: uses `/_stcore/health`; API uses `/health`.
- **nginx**: API routes are `/auth`, `/health` — not prefixed with `/api`.
