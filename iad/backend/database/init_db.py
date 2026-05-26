"""Schema bootstrap — Alembic is authoritative in production."""
from __future__ import annotations

# Import models so metadata is populated
from iad.backend import models as _models  # noqa: F401
from iad.backend.database.base import Base
from iad.backend.database.session import get_engine, reset_engine
from iad.config.settings import Settings, get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.database")


def create_all_tables(settings: Settings | None = None) -> None:
    """Create tables from ORM metadata (development / tests only).

    Production deployments MUST use ``alembic upgrade head`` instead.
    """
    cfg = settings or get_settings()
    engine = get_engine(cfg)
    Base.metadata.create_all(bind=engine)
    logger.info("database tables created via metadata.create_all")


def drop_all_tables(settings: Settings | None = None) -> None:
    """Drop all tables — **tests only**."""
    cfg = settings or get_settings()
    if cfg.ENVIRONMENT not in ("test", "development"):
        raise RuntimeError("drop_all_tables is only allowed in test/development")
    engine = get_engine(cfg)
    Base.metadata.drop_all(bind=engine)
    logger.warning("database tables dropped")


def init_database(settings: Settings | None = None, *, create: bool = False) -> None:
    """Initialise DB when ``AUTO_CREATE_DB`` or explicit ``create=True``."""
    cfg = settings or get_settings()
    if create or cfg.AUTO_CREATE_DB:
        create_all_tables(cfg)


def reset_database_for_tests() -> None:
    """Full reset between test modules."""
    reset_engine()
    get_settings.cache_clear()
