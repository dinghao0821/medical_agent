"""Database layer (P3) — SQLAlchemy engine/session with SQLite default.

Used for user accounts and audit logs. Defaults to a local SQLite file
(``data/app.db``) so the app runs with zero infra; set ``DATABASE_URL`` to a
PostgreSQL DSN for production. Initialized lazily via ``init_db(config)``.
"""

import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionLocal = None


def _database_url(config) -> str:
    url = getattr(getattr(config, "auth", None), "database_url", "") or ""
    return url or "sqlite:///./data/app.db"


def init_db(config) -> bool:
    """Initialize the engine + session factory and create tables. Idempotent.

    Returns True on success, False on failure (caller decides how to degrade).
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return True

    url = _database_url(config)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    try:
        if url.startswith("sqlite"):
            os.makedirs("data", exist_ok=True)
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        # Import models so their tables register on the shared Base metadata.
        from services import models  # noqa: F401
        Base.metadata.create_all(_engine)
        _migrate_add_columns(_engine, url)
        logger.info("Database initialized at %s", url)
        return True
    except Exception as e:
        logger.error("Database initialization failed (%s).", e)
        _engine = None
        _SessionLocal = None
        return False


def is_ready() -> bool:
    return _SessionLocal is not None


def get_session():
    """Return a new Session. Raises if the DB was not initialized."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized; call init_db(config) first.")
    return _SessionLocal()


# ---- Lightweight, idempotent schema migration ----------------------------
# ``create_all`` never ADDs columns to a table that already exists. To stay
# zero-downtime and backward compatible (existing SQLite/Postgres DBs), we add
# any missing columns for the doctor-licence-verification feature by hand.
_ADDITIVE_COLUMNS = {
    "users": [
        ("doctor_status", "VARCHAR(16) DEFAULT 'none'"),
        ("license_path", "VARCHAR(512)"),
        ("license_reviewed_by", "VARCHAR(64)"),
        ("license_reviewed_at", "TIMESTAMP"),
        ("license_comments", "TEXT"),
    ],
    "elderly_assessment_cases": [
        ("version", "INTEGER DEFAULT 1"),
        ("updated_at", "TIMESTAMP"),
    ],
}


def _existing_columns(engine, table):
    try:
        from sqlalchemy import inspect as _sa_inspect
        return {c["name"] for c in _sa_inspect(engine).get_columns(table)}
    except Exception:
        return set()


def _migrate_add_columns(engine, url):
    """Add missing additive columns; safe to run on every startup."""
    from sqlalchemy import text
    for table, cols in _ADDITIVE_COLUMNS.items():
        existing = _existing_columns(engine, table)
        if not existing:
            continue  # table not created yet / inspection failed
        for name, ddl in cols:
            if name in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                logger.info("Migrated: added %s.%s", table, name)
            except Exception as e:
                logger.warning("Skip adding column %s.%s: %s", table, name, e)
