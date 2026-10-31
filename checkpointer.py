"""Checkpointer factory supporting embedded SQLite (zero-ops default) and PostgreSQL.

When DB_URI is empty or starts with 'sqlite://', an embedded SQLite database is used
with Write-Ahead Logging (WAL) and busy timeout enabled for concurrency.
When DB_URI starts with 'postgresql://' or 'postgres://', PostgreSQL connection pooling
via psycopg-pool is used.

Checkpoint state is sealed at rest with EncryptedSerde (see vault.py) for both backends.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator

from langgraph.checkpoint.base import BaseCheckpointSaver

from config import DB_URI, ROOT
from vault import EncryptedSerde

logger = logging.getLogger("checkpointer")

_UUID6_EPOCH_OFFSET = 0x01B21DD213814000  # 100 ns ticks from 1582-10-15 to 1970-01-01


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


def checkpoint_time(checkpoint_id: str) -> float:
    """Unix time a checkpoint was written. LangGraph ids are UUID6, so the time is readable without decrypting."""
    digits = checkpoint_id.replace("-", "")
    ticks = int(digits[0:8] + digits[8:12] + digits[13:16], 16)
    return (ticks - _UUID6_EPOCH_OFFSET) / 1e7


@contextmanager
def get_checkpointer(
    uri: str | None = None,
    max_pool_size: int = 5,
) -> Generator[BaseCheckpointSaver, None, None]:
    """Context manager yielding an initialized, encrypting checkpointer ready for LangGraph."""
    target = uri if uri is not None else DB_URI

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        with ConnectionPool(conninfo=target, max_size=max_pool_size, open=True, kwargs={"autocommit": True}) as pool:
            saver = PostgresSaver(pool, serde=EncryptedSerde())  # type: ignore[arg-type]
            saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = get_sqlite_path(target)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn, serde=EncryptedSerde())
            saver.setup()
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
            except Exception as exc:
                logger.debug("SQLite PRAGMA tuning: %s", exc)
            yield saver
        finally:
            conn.close()


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
        db_path = get_sqlite_path(target)
        if not Path(db_path).exists():
            return []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id;")
            return [row[0] for row in cursor.fetchall()]


def prune_checkpoints(retention_days: int, uri: str | None = None, now: float | None = None) -> int:
    """Delete whole threads whose newest checkpoint is older than the retention period.

    Threads that are waiting for a human decision are never deleted, however old they are.
    Returns the number of threads removed. SQLite only; PostgreSQL retention is operator-run.
    """
    target = uri if uri is not None else DB_URI
    if is_postgres_uri(target):
        raise NotImplementedError("Checkpoint retention is implemented for SQLite. Apply it to PostgreSQL with your DBA tooling.")
    db_path = get_sqlite_path(target)
    if not Path(db_path).exists():
        return 0

    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    removed = 0
    with get_checkpointer(uri=target) as saver:
        conn = saver.conn
        latest = conn.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id").fetchall()
        for thread_id, newest_id in latest:
            if checkpoint_time(newest_id) >= cutoff:
                continue
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            snapshot = saver.get_tuple(config)
            if snapshot is not None and any(channel == "__interrupt__" for _, channel, _ in snapshot.pending_writes):
                continue
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            removed += 1
        conn.commit()
    return removed
