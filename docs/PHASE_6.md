# Phase 6 — Database & Storage

## Architecture

Phase 6 adds a **relational persistence layer** under `iad/backend/` using the repository pattern. Streamlit session state remains the primary runtime store; the database is **opt-in** (`IAD_DATABASE_ENABLED=false` by default) so the UI never requires PostgreSQL to start.

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit / FastAPI (Phase 5)                              │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
              PersistenceService (application layer)
                            ▼
                   UnitOfWork + Repositories
                            ▼
              SQLAlchemy ORM (iad.backend.models)
                            ▼
         PostgreSQL (prod)  /  SQLite (dev & tests)
```

### ADR-015 — SQLite default, PostgreSQL production

| Environment | URL |
|-------------|-----|
| Test | `sqlite:///:memory:` |
| Development | `sqlite:///{DATA_DIR}/iad.db` |
| Production | `IAD_DATABASE_URL=postgresql+psycopg://...` |

### ADR-016 — Repository + Unit of Work

Repositories encapsulate queries; `UnitOfWork` groups them in one transaction. FastAPI dependencies and `session_scope()` share the same session factory.

### Entity model

| Table | Purpose |
|-------|---------|
| `users` | Accounts (password field reserved for Phase 7) |
| `datasets` | Versioned dataset metadata + schema JSON |
| `experiments` | Training runs (status, config, MLflow id) |
| `ml_models` | Trained model registry + artifact path |
| `metrics` | Per-model / per-experiment metrics |
| `predictions` | Inference audit log |

Relationships use `ON DELETE CASCADE` for owned children; `dataset_id` on experiments is `SET NULL` when a dataset is removed.

## Directory tree

```
iad/backend/
├── database/
│   ├── base.py
│   ├── session.py
│   ├── init_db.py
│   └── migrations/
│       ├── env.py
│       └── versions/20260526_0001_initial_schema.py
├── models/
│   ├── user.py
│   ├── dataset.py
│   ├── experiment.py
│   ├── ml_model.py
│   ├── metric.py
│   └── prediction.py
├── repositories/
│   ├── base.py
│   ├── unit_of_work.py
│   └── …
└── services/
    └── persistence_service.py

alembic.ini
docker/docker-compose.postgres.yml
scripts/db_init.py
```

## Setup

### Local SQLite (zero config)

```bash
pip install -r requirements.txt
alembic upgrade head
# or: python scripts/db_init.py
```

### PostgreSQL

```bash
docker compose -f docker/docker-compose.postgres.yml up -d
export IAD_DATABASE_URL=postgresql+psycopg://iad:iad@localhost:5432/iad
alembic upgrade head
```

### Enable persistence from Streamlit

```env
IAD_DATABASE_ENABLED=true
IAD_AUTO_CREATE_DB=true   # dev only
```

## Integration (Phase 5 preview)

```python
from iad.backend.services import PersistenceService
from iad.backend.database.session import get_db

# FastAPI
@app.post("/datasets")
def register_dataset(..., db: Session = Depends(get_db)):
    ...

# After training
PersistenceService().persist_training_result(training_result, experiment_name="...")
```

## Testing

```bash
pytest tests/unit/backend/ -v
pytest --cov=iad.backend
```

Tests use in-memory SQLite via the `db_session` fixture.

## Rollback

```bash
alembic downgrade base   # or drop DB
git checkout HEAD -- iad/backend/database iad/backend/models iad/backend/repositories
rm -rf iad/backend/services/persistence_service.py alembic.ini
pip uninstall -y sqlalchemy alembic psycopg-binary
```

## Production considerations

- Run `alembic upgrade head` in CI/CD before app deploy — never `AUTO_CREATE_DB` in production.
- Use connection pooling settings (`DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`) for PostgreSQL.
- Store large parquet/CSV files on disk or S3; DB holds metadata + checksums only.
- Index strategy is included in the initial migration; add partial indexes when query patterns emerge.
