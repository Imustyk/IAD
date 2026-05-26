"""Engine and session factory — thread-safe session lifecycle."""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from iad.config.settings import Settings, get_settings
from iad.core.exceptions import DatabaseError
from iad.core.logging import get_logger

logger = get_logger("iad.database")

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _sqlite_pragmas(dbapi_conn: Any, _connection_record: Any) -> None:
    """Enable FK enforcement and WAL for file-backed SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def get_engine(settings: Settings | None = None) -> Engine:
    """Return a process-wide SQLAlchemy engine (cached)."""
    global _engine
    if _engine is not None:
        return _engine

    cfg = settings or get_settings()
    url = cfg.resolved_database_url()
    connect_args: dict[str, Any] = {}
    pool_kwargs: dict[str, Any] = {}

    pool_class = None
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in url:
            # Single shared in-memory DB across connections (tests + API).
            pool_class = StaticPool
    else:
        pool_kwargs = {
            "pool_size": cfg.DATABASE_POOL_SIZE,
            "max_overflow": cfg.DATABASE_MAX_OVERFLOW,
            "pool_pre_ping": True,
        }

    engine_kwargs: dict[str, Any] = {
        "echo": cfg.DATABASE_ECHO,
        "connect_args": connect_args,
        **pool_kwargs,
    }
    if pool_class is not None:
        engine_kwargs["poolclass"] = pool_class
    _engine = create_engine(url, **engine_kwargs)

    if url.startswith("sqlite") and ":memory:" not in url:
        event.listen(_engine, "connect", _sqlite_pragmas)

    logger.info("database engine initialised", extra={"db_url_scheme": url.split("://")[0]})
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return a cached sessionmaker bound to :func:`get_engine`."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal

    _SessionLocal = sessionmaker(
        bind=get_engine(settings),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    return _SessionLocal


def reset_engine() -> None:
    """Dispose engine and bust caches — for tests and settings reload."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
@contextmanager
def session_scope(settings: Settings | None = None) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.exception("database transaction rolled back")
        raise DatabaseError(
            str(exc),
            user_message="A database transaction failed.",
        ) from exc
    finally:
        session.close()


def get_db(settings: Settings | None = None) -> Generator[Session, None, None]:
    """FastAPI-style dependency generator yielding a DB session."""
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(settings: Settings | None = None) -> bool:
    """Return True if a simple ``SELECT 1`` succeeds."""
    try:
        with get_engine(settings).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("database health check failed: %s", exc)
        return False
