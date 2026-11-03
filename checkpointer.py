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
from contextlib import closing, contextmanager
from pathlib import Path
from collections.abc import Generator

from langgraph.checkpoint.base import BaseCheckpointSaver

from config import DB_URI, ROOT
from vault import EncryptedSerde, bind_thread

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


def _thread_of(config) -> str | None:
    thread = ((config or {}).get("configurable") or {}).get("thread_id")
    return None if thread is None else str(thread)


class _ThreadBound:
    """Mixin: every saver call runs with its thread bound, so sealed state only opens for that thread."""

    def put(self, config, *args, **kwargs):
        with bind_thread(_thread_of(config) or ""):
            return super().put(config, *args, **kwargs)  # type: ignore[misc]

    def put_writes(self, config, *args, **kwargs):
        with bind_thread(_thread_of(config) or ""):
            return super().put_writes(config, *args, **kwargs)  # type: ignore[misc]

    def get_tuple(self, config):
        with bind_thread(_thread_of(config) or ""):
            return super().get_tuple(config)  # type: ignore[misc]

    def list(self, config, *args, **kwargs):
        thread = _thread_of(config)
        if thread is None:
            yield from super().list(config, *args, **kwargs)  # type: ignore[misc]
            return
        with bind_thread(thread):
            yield from super().list(config, *args, **kwargs)  # type: ignore[misc]


def _bound_sqlite_saver(conn: sqlite3.Connection) -> BaseCheckpointSaver:
    from langgraph.checkpoint.sqlite import SqliteSaver

    class BoundSqliteSaver(_ThreadBound, SqliteSaver):
        pass

    return BoundSqliteSaver(conn, serde=EncryptedSerde())


def _bound_postgres_saver(pool) -> BaseCheckpointSaver:
    from langgraph.checkpoint.postgres import PostgresSaver

    class BoundPostgresSaver(_ThreadBound, PostgresSaver):
        pass

    return BoundPostgresSaver(pool, serde=EncryptedSerde())  # type: ignore[arg-type]


@contextmanager
def get_checkpointer(
    uri: str | None = None,
    max_pool_size: int = 5,
) -> Generator[BaseCheckpointSaver, None, None]:
    """Context manager yielding an initialized, encrypting checkpointer ready for LangGraph."""
    target = uri if uri is not None else DB_URI

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool

        with ConnectionPool(conninfo=target, max_size=max_pool_size, open=True, kwargs={"autocommit": True}) as pool:
            saver = _bound_postgres_saver(pool)
            saver.setup()  # type: ignore[attr-defined]
            yield saver
    else:
        db_path = get_sqlite_path(target)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            saver = _bound_sqlite_saver(conn)
            saver.setup()
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
            except Exception as exc:
                logger.debug("SQLite PRAGMA tuning: %s", exc)
            yield saver
        finally:
            conn.close()


def fetch_checkpoint_threads(
    uri: str | None = None,
    prefix: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Distinct thread IDs in the checkpointer database.

    With a prefix, only matching threads are returned. With a limit, the most recently active
    threads come first (checkpoint IDs are time-ordered), so callers do not decrypt every thread.
    Without either, IDs are sorted alphabetically as before.
    """
    target = uri if uri is not None else DB_URI
    escaped = None
    if prefix:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    where = " WHERE thread_id LIKE %s ESCAPE '\\'" if escaped else ""
    order = " ORDER BY MAX(checkpoint_id) DESC" if limit else " ORDER BY thread_id"
    tail = f" LIMIT {int(limit)}" if limit else ""

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool

        query = f"SELECT thread_id FROM checkpoints{where} GROUP BY thread_id{order}{tail};"
        with ConnectionPool(conninfo=target, min_size=1, max_size=1, open=True, kwargs={"autocommit": True}) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (escaped,) if escaped else None)
                    return [row[0] for row in cur.fetchall()]

    db_path = get_sqlite_path(target)
    if not Path(db_path).exists():
        return []
    query = f"SELECT thread_id FROM checkpoints{where.replace('%s', '?')} GROUP BY thread_id{order}{tail};"
    with closing(sqlite3.connect(db_path)) as conn:
        try:
            cursor = conn.execute(query, (escaped,) if escaped else ())
        except sqlite3.OperationalError:
            return []
        return [row[0] for row in cursor.fetchall()]


def _is_waiting_for_review(saver: BaseCheckpointSaver, thread_id: str) -> bool:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    snapshot = saver.get_tuple(config)
    return snapshot is not None and any(channel == "__interrupt__" for _, channel, _ in snapshot.pending_writes)


def prune_checkpoints(retention_days: int, uri: str | None = None, now: float | None = None) -> int:
    """Delete whole threads whose newest checkpoint is older than the retention period.

    Threads that are waiting for a human decision are never deleted, however old they are.
    Returns the number of threads removed. Works for SQLite and PostgreSQL.
    """
    target = uri if uri is not None else DB_URI
    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    removed = 0

    if is_postgres_uri(target):
        from psycopg_pool import ConnectionPool

        with get_checkpointer(uri=target) as saver:
            pool = saver.conn  # type: ignore[attr-defined]
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id")
                    latest = cur.fetchall()
            for thread_id, newest_id in latest:
                if checkpoint_time(newest_id) >= cutoff or _is_waiting_for_review(saver, thread_id):
                    continue
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                            cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
                removed += 1
        return removed

    db_path = get_sqlite_path(target)
    if not Path(db_path).exists():
        return 0
    with get_checkpointer(uri=target) as saver:
        conn = saver.conn  # type: ignore[attr-defined]
        latest = conn.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id").fetchall()
        for thread_id, newest_id in latest:
            if checkpoint_time(newest_id) >= cutoff or _is_waiting_for_review(saver, thread_id):
                continue
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            removed += 1
        conn.commit()
    return removed
