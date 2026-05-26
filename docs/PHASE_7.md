# Phase 7 — Authentication & Security

## Architecture

Phase 7 adds **defense in depth**: password hashing, JWT sessions, RBAC, CSRF for cookie clients, rate limiting, secure HTTP headers, and centralised upload policy. Streamlit auth is **opt-in** (`IAD_AUTH_ENABLED=false` by default).

```
┌──────────────┐     Bearer JWT      ┌─────────────────────┐
│  Streamlit   │◄───────────────────►│  FastAPI /auth/*    │
│  (optional)  │                     │  + middleware stack │
└──────┬───────┘                     └──────────┬──────────┘
       │                                        │
       └────────────────┬───────────────────────┘
                        ▼
              AuthService → UnitOfWork → users
                        ▼
              bcrypt / jose / RBAC / CSRF
```

### ADR-017 — bcrypt directly (not passlib)

passlib is unmaintained and breaks on bcrypt 4.2+. We use the `bcrypt` library with explicit cost factor 12.

### ADR-018 — JWT access + refresh

| Token | TTL | Use |
|-------|-----|-----|
| access | 30 min (configurable) | API `Authorization: Bearer` |
| refresh | 7 days | `/auth/refresh` only |

Each token carries `jti` for uniqueness and future revocation lists.

### ADR-019 — RBAC roles

| Role | Permissions |
|------|-------------|
| viewer | read datasets, experiments |
| analyst | + write, train, predict |
| admin | all + user admin |

`is_superuser` maps to `admin`.

## Directory tree

```
iad/backend/security/
├── passwords.py
├── jwt_tokens.py
├── permissions.py
├── csrf.py
├── rate_limit.py
└── upload_policy.py

iad/backend/services/auth_service.py
iad/backend/schemas/auth.py
iad/backend/middleware/
iad/backend/api/
├── app.py
├── deps.py
└── routes/auth.py, health.py

iad/frontend/auth/gate.py
scripts/run_api.py
scripts/create_admin.py
```

## API endpoints

| Method | Path | Auth |
|--------|------|------|
| POST | `/auth/register` | Public |
| POST | `/auth/login` | Public |
| POST | `/auth/refresh` | Refresh token body |
| GET | `/auth/me` | Bearer |
| POST | `/auth/change-password` | Bearer |
| GET | `/health`, `/healthz` | Public |

Run API:

```bash
alembic upgrade head
python scripts/create_admin.py --email admin@iad.local --password 'AdminPass123'
python scripts/run_api.py --reload
```

## Streamlit auth

```env
IAD_AUTH_ENABLED=true
IAD_AUTO_CREATE_DB=true
```

Sign-in / register tabs appear before the dashboard. Session stores `user` and `auth_tokens`.

## Testing

```bash
pytest tests/unit/backend/test_security.py tests/unit/backend/test_auth_service.py -v
pytest tests/integration/test_api_auth.py -v
```

## Rollback

```bash
git checkout HEAD -- iad/backend/security iad/backend/api iad/backend/middleware
git checkout HEAD -- iad/frontend/auth iad/backend/services/auth_service.py
pip uninstall -y fastapi uvicorn bcrypt python-jose
```

## Production checklist

- Set `IAD_SECRET_KEY` to 64+ random bytes
- Set `IAD_ENVIRONMENT=production`
- Disable `IAD_DEBUG` and restrict CORS origins
- Use PostgreSQL + Alembic migrations
- Put API behind TLS (nginx / load balancer)
- Replace in-memory rate limiter with Redis for multi-replica deployments
