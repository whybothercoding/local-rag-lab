"""Checkpointer factory supporting embedded SQLite (zero-ops default) and PostgreSQL.

When DB_URI is empty or starts with 'sqlite://', an embedded SQLite database is used
with Write-Ahead Logging (WAL) and busy timeout enabled for concurrency.
When DB_URI starts with 'postgresql://' or 'postgres://', PostgreSQL connection pooling
via psycopg-pool is used.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator

from langgraph.checkpoint.base import BaseCheckpointSaver

from config import DB_URI, ROOT

logger = logging.getLogger("checkpointer")


def is_postgres_uri(uri: str | None = None) -> bool:
    """Return True if the URI specifies a PostgreSQL connection."""
    target = uri if uri is not None else DB_URI
    return target.startswith(("postgresql://", "postgres://"))


def get_sqlite_path(uri: str | None = None) -> str:
    """Resolve and ensure parent directories for SQLite checkpoint file."""
    target = uri if uri is not None else DB_URI
    if target.startswith("sqlite:///"):
        raw_path = target[len("sqlite:///"):]
    elif target.startswith("sqlite://"):
        raw_path = target[len("sqlite://"):]
    else:
        raw_path = str(ROOT / "data" / "checkpoints.db")

    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def get_checkpointer(
    uri: str | None = None,
    max_pool_size: int = 5,
) -> Generator[BaseCheckpointSaver, None, None]:
    """Context manager yielding an initialized checkpointer ready for LangGraph.

    Automatically handles setup() and WAL mode for SQLite,
    or connection pool management for PostgreSQL.
    """
    target = uri if uri is not None else DB_URI

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        with ConnectionPool(conninfo=target, max_size=max_pool_size, open=True, kwargs={"autocommit": True}) as pool:
            saver = PostgresSaver(pool)  # type: ignore[arg-type]
            saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = get_sqlite_path(target)
        with SqliteSaver.from_conn_string(db_path) as saver:
            saver.setup()
            try:
                saver.conn.execute("PRAGMA journal_mode=WAL;")
                saver.conn.execute("PRAGMA busy_timeout=5000;")
            except Exception as exc:
                logger.debug("SQLite PRAGMA tuning: %s", exc)
            yield saver


def fetch_checkpoint_threads(uri: str | None = None) -> list[str]:
    """Retrieve distinct thread IDs stored in the checkpointer database."""
    target = uri if uri is not None else DB_URI

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool
        query = "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id;"
        with ConnectionPool(conninfo=target, min_size=1, max_size=1, open=True, kwargs={"autocommit": True}) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return [row[0] for row in cur.fetchall()]
    else:
        import sqlite3
        db_path = get_sqlite_path(target)
        if not Path(db_path).exists():
            return []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id;")
            return [row[0] for row in cursor.fetchall()]
