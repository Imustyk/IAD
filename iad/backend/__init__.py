"""FastAPI backend layer — persistence (Phase 6), API routes (Phase 5), security (Phase 7).

Subpackages
-----------
* ``iad.backend.database``    — engine, sessions, Alembic migrations
* ``iad.backend.models``      — SQLAlchemy ORM entities
* ``iad.backend.repositories`` — repository pattern + Unit of Work
* ``iad.backend.services``    — use-case orchestration (persistence, training API)
* ``iad.backend.api``         — FastAPI routers (Phase 5)
* ``iad.backend.security``    — JWT, password hashing (Phase 7)
"""
from iad.backend.services.auth_service import AuthService
from iad.backend.services.persistence_service import PersistenceService

__all__ = ["AuthService", "PersistenceService"]
