#!/usr/bin/env python3
"""Append-only, hash-chained audit trail for queries, groundedness checks, and human decisions.

Each record stores the SHA-256 of the record before it, so changing or removing any row
breaks verification from that point on. Triggers also reject UPDATE and DELETE.

Writes are fail-closed: if an event cannot be stored, the calling node raises and no
answer is released.

Limitation: the chain cannot detect truncation of the newest records by someone who
controls the database file. Export `head` periodically to an external store to close that gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import AUDIT_DB_PATH

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
"""

_COLUMNS = "seq, ts, event_type, workspace_id, session_id, data, prev_hash, record_hash"


def _record_digest(prev_hash: str, ts: str, event_type: str, workspace_id: str, session_id: str, data: str) -> str:
    body = json.dumps([prev_hash, ts, event_type, workspace_id, session_id, data], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


_schema_lock = threading.Lock()


def _open(path: str | None) -> sqlite3.Connection:
    target = Path(path or AUDIT_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000;")
    # Schema setup takes locks that can fail with SQLITE_BUSY, so it runs only when the table is missing.
    if not _table_exists(conn):
        with _schema_lock:
            if not _table_exists(conn):
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.executescript(_SCHEMA)
    return conn


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'audit_events'").fetchone()
    return row is not None


def record(
    event_type: str,
    *,
    workspace_id: str,
    session_id: str,
    data: dict[str, Any],
    path: str | None = None,
) -> int:
    """Append one event and return its sequence number. Raises if it cannot be stored."""
    ts = datetime.now(UTC).isoformat(timespec="microseconds")
    data_json = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    conn = _open(path)
    try:
        # BEGIN IMMEDIATE takes the write lock before reading the head, so concurrent
        # writers cannot both chain onto the same previous record.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash = row[0] if row else GENESIS_HASH
        digest = _record_digest(prev_hash, ts, event_type, workspace_id, session_id, data_json)
        cursor = conn.execute(
            "INSERT INTO audit_events (ts, event_type, workspace_id, session_id, data, prev_hash, record_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, event_type, workspace_id, session_id, data_json, prev_hash, digest),
        )
        conn.execute("COMMIT")
        if cursor.lastrowid is None:
            raise RuntimeError("audit insert returned no row id")
        return cursor.lastrowid
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def verify_chain(path: str | None = None) -> tuple[bool, int | None, int]:
    """Recompute every hash. Returns (ok, first_bad_seq, record_count)."""
    conn = _open(path)
    try:
        prev_hash = GENESIS_HASH
        count = 0
        rows = conn.execute(f"SELECT {_COLUMNS} FROM audit_events ORDER BY seq")
        for seq, ts, event_type, workspace_id, session_id, data, stored_prev, stored_hash in rows:
            count += 1
            expected = _record_digest(prev_hash, ts, event_type, workspace_id, session_id, data)
            if stored_prev != prev_hash or stored_hash != expected:
                return False, seq, count
            prev_hash = stored_hash
        return True, None, count
    finally:
        conn.close()


def head_hash(path: str | None = None) -> str:
    """Hash of the newest record. Store it outside the database to detect truncation."""
    conn = _open(path)
    try:
        row = conn.execute("SELECT record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS_HASH
    finally:
        conn.close()


def list_events(limit: int = 50, workspace_id: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
    """Newest-first events, optionally filtered to one workspace."""
    conn = _open(path)
    try:
        sql = f"SELECT {_COLUMNS} FROM audit_events"
        params: list[Any] = []
        if workspace_id:
            sql += " WHERE workspace_id = ?"
            params.append(workspace_id)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "seq": seq,
            "ts": ts,
            "event_type": event_type,
            "workspace_id": workspace_id_,
            "session_id": session_id,
            "data": json.loads(data),
            "record_hash": record_hash,
        }
        for seq, ts, event_type, workspace_id_, session_id, data, _prev, record_hash in rows
    ]


def list_rejections(limit: int = 50, workspace_id: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
    """Newest-first human rejections with the reviewer's reason. Feedback on answers the reviewer declined to release."""
    conn = _open(path)
    try:
        sql = f"SELECT {_COLUMNS} FROM audit_events WHERE event_type = 'hitl_decision' AND json_extract(data, '$.released') = 0"
        params: list[Any] = []
        if workspace_id:
            sql += " AND workspace_id = ?"
            params.append(workspace_id)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "seq": seq,
            "ts": ts,
            "workspace_id": workspace_id_,
            "session_id": session_id,
            "data": json.loads(data),
        }
        for seq, ts, _event, workspace_id_, session_id, data, _prev, _hash in rows
    ]


def count_events_since(
    event_type: str,
    json_field: str,
    value: str,
    since: datetime,
    path: str | None = None,
) -> int:
    """Count events of one type whose data field matches a value, recorded at or after a point in time."""
    conn = _open(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ? AND json_extract(data, ?) = ? AND ts >= ?",
            (event_type, f"$.{json_field}", value, since.isoformat(timespec="microseconds")),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and inspect the audit trail.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Recompute the hash chain and report the first broken record")
    sub.add_parser("head", help="Print the newest record hash for external anchoring")
    feedback = sub.add_parser("feedback", help="List answers reviewers rejected, with their reasons")
    feedback.add_argument("-n", type=int, default=20)
    feedback.add_argument("--workspace")
    tail = sub.add_parser("tail", help="Show recent events as JSON lines")
    tail.add_argument("-n", type=int, default=20, help="Number of events to show")
    tail.add_argument("--workspace", help="Only show events for this workspace")
    args = parser.parse_args(argv)

    if args.command == "verify":
        ok, bad_seq, count = verify_chain()
        if ok:
            print(f"OK: {count} record(s) verified. Head: {head_hash()}")
            return 0
        print(f"FAILED: chain breaks at seq {bad_seq} (checked {count} record(s)).", file=sys.stderr)
        return 1

    if args.command == "head":
        print(head_hash())
        return 0

    if args.command == "feedback":
        rejections = list_rejections(limit=args.n, workspace_id=args.workspace)
        if not rejections:
            print("No rejected answers recorded.")
        for item in rejections:
            data = item["data"]
            print(f"{item['ts']} [{item['workspace_id']}] approver={data.get('approver') or '-'}")
            print(f"  reason: {data.get('reason') or '(none given)'}")
            print(f"  groundedness: {data.get('groundedness_ratio')}  degraded: {data.get('degraded') or 'no'}")
            print(f"  citations: {', '.join(data.get('citations', [])) or 'none'}")
        return 0

    for event in reversed(list_events(limit=args.n, workspace_id=args.workspace)):
        print(json.dumps(event, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
