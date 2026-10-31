#!/usr/bin/env python3
"""Append-only, hash-chained, encrypted audit trail.

Each record stores the SHA-256 of the record before it, so changing or removing any row breaks
verification from that point on. Triggers also reject UPDATE and DELETE. Payloads are sealed
(AES-256-GCM, see vault.py) and bound to their row metadata.

Three limits are handled explicitly:
- Truncation of the newest records cannot be seen from the chain alone. Anchors record the head
  hash in a separate, signed file; verification checks every anchor against the log.
- Retention deletes old records. The deletion is recorded as a chain event, and the chain keeps the
  hash of the last deleted record as its base, so the remaining log still verifies.
- Writes are fail-closed: if an event cannot be stored, the calling node raises and no answer is released.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sqlite3
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from config import ANCHOR_PATH, AUDIT_DB_PATH, RETENTION_DAYS
from vault import anchor_mac, chain_mac, open_text, seal_text, subject_tag

GENESIS_HASH = "0" * 64
_schema_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
"""

_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
"""

_COLUMNS = "seq, ts, event_type, workspace_id, session_id, subject, data, prev_hash, record_hash"


def _digest(prev_hash: str, ts: str, event_type: str, workspace_id: str, session_id: str,
            subject: str, stored_data: str) -> str:
    body = json.dumps(
        [prev_hash, ts, event_type, workspace_id, session_id, subject, stored_data],
        separators=(",", ":"),
    )
    return chain_mac(body)


def _aad(ts: str, event_type: str, workspace_id: str, session_id: str, subject: str) -> bytes:
    return "|".join([ts, event_type, workspace_id, session_id, subject]).encode("utf-8")


def _schema_state(conn: sqlite3.Connection) -> tuple[set[str], set[str], set[str]]:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(audit_events)")}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    return columns, tables, triggers


def _schema_complete(columns: set[str], tables: set[str], triggers: set[str]) -> bool:
    return (
        {"audit_events", "audit_meta"} <= tables
        and "subject" in columns
        and {"audit_events_no_update", "audit_events_no_delete"} <= triggers
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    if _schema_complete(*_schema_state(conn)):
        return
    with _schema_lock:
        # Re-read under the lock: another thread may have finished the migration while we waited.
        columns, tables, _triggers = _schema_state(conn)
        if _schema_complete(*_schema_state(conn)):
            return
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)
        if "audit_events" in tables and "subject" not in columns:
            conn.execute("ALTER TABLE audit_events ADD COLUMN subject TEXT NOT NULL DEFAULT ''")
        conn.executescript(_NO_UPDATE_TRIGGER + _NO_DELETE_TRIGGER)


def _open(path: str | None = None) -> sqlite3.Connection:
    target = Path(path or AUDIT_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000;")
    _ensure_schema(conn)
    return conn


def _base(conn: sqlite3.Connection) -> tuple[int, str]:
    """The last record deleted by retention, and its hash. The chain starts from it."""
    rows = dict(conn.execute("SELECT key, value FROM audit_meta WHERE key IN ('base_seq', 'base_hash')").fetchall())
    return int(rows.get("base_seq", 0)), rows.get("base_hash", GENESIS_HASH)


def _decode(row: tuple) -> dict[str, Any]:
    seq, ts, event_type, workspace_id, session_id, subject, stored, _prev, record_hash = row
    try:
        data = json.loads(open_text(stored, _aad(ts, event_type, workspace_id, session_id, subject)))
    except Exception:
        data = {"_unreadable": "payload failed decryption; see audit.py verify"}
    return {
        "seq": seq,
        "ts": ts,
        "event_type": event_type,
        "workspace_id": workspace_id,
        "session_id": session_id,
        "data": data,
        "record_hash": record_hash,
    }


def record(
    event_type: str,
    *,
    workspace_id: str,
    session_id: str,
    data: dict[str, Any],
    subject: str = "",
    path: str | None = None,
) -> int:
    """Append one event and return its sequence number. Raises if it cannot be stored."""
    ts = datetime.now(UTC).isoformat(timespec="microseconds")
    subject_value = subject_tag(subject) if subject else ""
    stored = seal_text(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str),
        _aad(ts, event_type, workspace_id, session_id, subject_value),
    )
    conn = _open(path)
    try:
        # BEGIN IMMEDIATE takes the write lock before reading the head, so concurrent
        # writers cannot both chain onto the same previous record.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash = row[0] if row else _base(conn)[1]
        digest = _digest(prev_hash, ts, event_type, workspace_id, session_id, subject_value, stored)
        cursor = conn.execute(
            "INSERT INTO audit_events (ts, event_type, workspace_id, session_id, subject, data, prev_hash, record_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, event_type, workspace_id, session_id, subject_value, stored, prev_hash, digest),
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
    """Recompute every hash from the retention base. Returns (ok, first_bad_seq, record_count)."""
    conn = _open(path)
    try:
        _base_seq, prev_hash = _base(conn)
        count = 0
        for row in conn.execute(f"SELECT {_COLUMNS} FROM audit_events ORDER BY seq"):
            seq, ts, event_type, workspace_id, session_id, subject, stored, stored_prev, stored_hash = row
            count += 1
            expected = _digest(prev_hash, ts, event_type, workspace_id, session_id, subject, stored)
            if stored_prev != prev_hash or stored_hash != expected:
                return False, seq, count
            prev_hash = stored_hash
        return True, None, count
    finally:
        conn.close()


def head_hash(path: str | None = None) -> str:
    conn = _open(path)
    try:
        row = conn.execute("SELECT record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else _base(conn)[1]
    finally:
        conn.close()


def list_events(
    limit: int = 50,
    workspace_id: str | None = None,
    event_type: str | None = None,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Newest-first events, decrypted, optionally filtered."""
    conn = _open(path)
    try:
        clauses, params = [], []
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM audit_events{where} ORDER BY seq DESC LIMIT ?", [*params, limit]
        ).fetchall()
    finally:
        conn.close()
    return [_decode(row) for row in rows]


def list_rejections(limit: int = 50, workspace_id: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
    """Newest-first human rejections with the reviewer's reason. Feedback on answers the reviewer declined to release."""
    candidates = list_events(limit=10_000, workspace_id=workspace_id, event_type="hitl_decision", path=path)
    rejected = [event for event in candidates if event["data"].get("released") is False]
    return rejected[:limit]


def export_events(path: str | None = None) -> list[dict[str, Any]]:
    """Every event, oldest first, decrypted. For auditors. Exporting is itself audited by the caller."""
    return list(reversed(list_events(limit=10**9, path=path)))


def count_events_since(event_type: str, subject: str, since: datetime, path: str | None = None) -> int:
    """Count events of one type for one subject (keyed tag, never the plain name), recorded at or after a point in time."""
    conn = _open(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ? AND subject = ? AND ts >= ?",
            (event_type, subject_tag(subject), since.isoformat(timespec="microseconds")),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


# --- Anchors ----------------------------------------------------------------

def _read_anchors(anchor_path: str | None) -> list[dict[str, Any]]:
    target = Path(anchor_path or ANCHOR_PATH)
    if not target.exists():
        return []
    with open(target, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def anchor(path: str | None = None, anchor_path: str | None = None) -> dict[str, Any] | None:
    """Record the current head hash in the anchor file. Each anchor is signed and chained to the previous one."""
    conn = _open(path)
    try:
        head = conn.execute("SELECT seq, record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if head is None:
        return None
    anchors = _read_anchors(anchor_path)
    entry: dict[str, Any] = {
        "seq": head[0],
        "hash": head[1],
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "prev": anchors[-1]["mac"] if anchors else GENESIS_HASH,
    }
    entry["mac"] = anchor_mac(_canonical(entry))
    target = Path(anchor_path or ANCHOR_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify_anchors(path: str | None = None, anchor_path: str | None = None) -> tuple[bool, list[str], int]:
    """Check every anchor's signature and chaining, and that the anchored records still exist unchanged.

    Anchors older than the retention base describe records that were deleted on purpose; they are
    accepted without a record to compare against.
    """
    problems: list[str] = []
    anchors = _read_anchors(anchor_path)
    conn = _open(path)
    try:
        base_seq, base_hash = _base(conn)
        prev = GENESIS_HASH
        for index, entry in enumerate(anchors, 1):
            body = {key: value for key, value in entry.items() if key != "mac"}
            if not hmac.compare_digest(anchor_mac(_canonical(body)), str(entry.get("mac", ""))):
                problems.append(f"anchor {index}: signature does not match (anchor file was edited)")
            if entry.get("prev") != prev:
                problems.append(f"anchor {index}: chaining to the previous anchor is broken")
            prev = str(entry.get("mac", ""))

            seq = int(entry["seq"])
            row = conn.execute("SELECT record_hash FROM audit_events WHERE seq = ?", (seq,)).fetchone()
            if row is not None:
                if row[0] != entry["hash"]:
                    problems.append(f"anchor {index}: record {seq} changed after it was anchored")
            elif seq == base_seq:
                if base_hash != entry["hash"]:
                    problems.append(f"anchor {index}: retention base does not match the anchored hash")
            elif seq > base_seq:
                problems.append(
                    f"anchor {index}: anchored record {seq} is missing; the log was truncated"
                )
    finally:
        conn.close()
    return not problems, problems, len(anchors)


# --- Retention --------------------------------------------------------------

def prune(retention_days: int = RETENTION_DAYS, path: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Delete records older than the retention period. The deletion is recorded in the chain.

    The chain base moves to the hash of the last deleted record, so the remaining log still verifies.
    """
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=retention_days)).isoformat(timespec="microseconds")
    conn = _open(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        last = conn.execute(
            "SELECT seq, record_hash FROM audit_events WHERE ts < ? ORDER BY seq DESC LIMIT 1", (cutoff,)
        ).fetchone()
        if last is None:
            conn.execute("COMMIT")
            return {"cutoff": cutoff, "removed": 0, "base_seq": _base(conn)[0]}
        last_seq, last_hash = last
        removed = conn.execute("SELECT COUNT(*) FROM audit_events WHERE seq <= ?", (last_seq,)).fetchone()[0]
        conn.execute("DROP TRIGGER audit_events_no_delete")
        conn.execute("DELETE FROM audit_events WHERE seq <= ?", (last_seq,))
        conn.execute(_NO_DELETE_TRIGGER)
        conn.execute("INSERT OR REPLACE INTO audit_meta (key, value) VALUES ('base_seq', ?)", (str(last_seq),))
        conn.execute("INSERT OR REPLACE INTO audit_meta (key, value) VALUES ('base_hash', ?)", (last_hash,))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    record(
        "retention_prune",
        workspace_id="",
        session_id="",
        data={"cutoff": cutoff, "removed": removed, "base_seq": last_seq, "retention_days": retention_days},
        path=path,
    )
    return {"cutoff": cutoff, "removed": removed, "base_seq": last_seq}


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify, anchor, and inspect the audit trail.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Recompute the hash chain and check every anchor")
    sub.add_parser("anchor", help="Record the current head hash in the signed anchor file")
    sub.add_parser("head", help="Print the newest record hash for external anchoring")
    tail = sub.add_parser("tail", help="Show recent events as JSON lines")
    tail.add_argument("-n", type=int, default=20, help="Number of events to show")
    tail.add_argument("--workspace", help="Only show events for this workspace")
    feedback = sub.add_parser("feedback", help="List answers reviewers rejected, with their reasons")
    feedback.add_argument("-n", type=int, default=20)
    feedback.add_argument("--workspace")
    sub.add_parser("export", help="Print every event, oldest first, as JSON lines (decrypted)")
    args = parser.parse_args(argv)

    if args.command == "verify":
        ok, bad_seq, count = verify_chain()
        anchors_ok, problems, anchor_count = verify_anchors()
        if ok:
            print(f"Chain OK: {count} record(s). Head: {head_hash()}")
        else:
            print(f"Chain FAILED at seq {bad_seq} (checked {count} record(s)).", file=sys.stderr)
        print(f"Anchors: {anchor_count} checked.")
        for problem in problems:
            print(f"Anchors FAILED: {problem}", file=sys.stderr)
        return 0 if ok and anchors_ok else 1

    if args.command == "anchor":
        entry = anchor()
        print("Nothing to anchor: the log is empty." if entry is None else f"Anchored seq {entry['seq']}: {entry['hash']}")
        return 0

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

    if args.command == "export":
        for event in export_events():
            _print_event(event)
        return 0

    for event in reversed(list_events(limit=args.n, workspace_id=args.workspace)):
        _print_event(event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
