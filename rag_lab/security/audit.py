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
import hmac
import json
import os
import sqlite3
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rag_lab.config import settings
from rag_lab.retrieval.citation import source_file
from rag_lab.security.vault import anchor_mac, base_mac, chain_mac, open_text, seal_text, subject_tag

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
            try:
                conn.execute("ALTER TABLE audit_events ADD COLUMN subject TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                # Two processes first-opening the same database can race past the thread-only lock
                # above; the schema re-check decides whether the other one already did our work.
                if "subject" not in _schema_state(conn)[0]:
                    raise
        conn.executescript(_NO_UPDATE_TRIGGER + _NO_DELETE_TRIGGER)


def _open(path: str | None = None) -> sqlite3.Connection:
    target = Path(path or settings.audit_db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000;")
    _ensure_schema(conn)
    return conn


def _base(conn: sqlite3.Connection) -> tuple[int, str]:
    """The last record deleted by retention, and its hash. The chain starts from it."""
    rows = dict(conn.execute("SELECT key, value FROM audit_meta WHERE key IN ('base_seq', 'base_hash')").fetchall())
    return int(rows.get("base_seq", 0)), rows.get("base_hash", GENESIS_HASH)


def _base_problems(conn: sqlite3.Connection) -> list[str]:
    """Problems with the retention base. audit_meta carries no append-only triggers, so the base is
    integrity-tagged with a keyed MAC and cross-checked against the prune event the chain must hold.

    This is what defeats a truncation cover-up: deleting the records up to an anchored seq and writing
    that seq/hash into audit_meta makes the remaining chain verify, but the base tag cannot be forged
    without the data key, and a legitimate prune always records a sealed retention_prune event."""
    rows = dict(conn.execute("SELECT key, value FROM audit_meta").fetchall())
    try:
        base_seq = int(rows.get("base_seq", 0))
    except (TypeError, ValueError):
        return ["retention base seq is not a number (audit_meta was edited)"]
    if base_seq <= 0:
        return []
    base_hash = rows.get("base_hash", "")
    problems: list[str] = []
    body = json.dumps([str(base_seq), base_hash], separators=(",", ":"))
    tag = str(rows.get("base_mac", ""))
    if not tag or not hmac.compare_digest(base_mac(body), tag):
        problems.append(
            f"retention base (seq {base_seq}) does not match its integrity tag (audit_meta was edited, "
            "or the base predates integrity tags)"
        )
    prune_rows = conn.execute(
        f"SELECT {_COLUMNS} FROM audit_events WHERE event_type = 'retention_prune' ORDER BY seq"
    ).fetchall()
    if not any(_decode(row)["data"].get("base_seq") == base_seq for row in prune_rows):
        problems.append(
            f"retention base (seq {base_seq}) has no matching retention_prune event in the log; "
            "records were deleted outside retention"
        )
    return problems


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
        if cursor.lastrowid is None:
            # Checked before the COMMIT: a row that stored but was reported as failed would invert the
            # fail-closed contract (the answer is withheld although the event exists).
            raise RuntimeError("audit insert returned no row id")
        conn.execute("COMMIT")
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


# The `released` flag lives inside the sealed payload, so candidates cannot be filtered in SQL;
# the scan pages backwards through the whole history instead of stopping at a fixed cap, so no
# rejection becomes unreachable no matter how many decisions came after it.
_REJECTION_PAGE = 2000


def list_rejections(limit: int = 50, workspace_id: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
    """Newest-first human rejections with the reviewer's reason. Feedback on answers the reviewer declined to release."""
    rejected: list[dict[str, Any]] = []
    last_seq: int | None = None
    conn = _open(path)
    try:
        while len(rejected) < limit:
            clauses = ["event_type = 'hitl_decision'"]
            params: list[Any] = []
            if workspace_id:
                clauses.append("workspace_id = ?")
                params.append(workspace_id)
            if last_seq is not None:
                clauses.append("seq < ?")
                params.append(last_seq)
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM audit_events WHERE {' AND '.join(clauses)} ORDER BY seq DESC LIMIT {_REJECTION_PAGE}",
                params,
            ).fetchall()
            if not rows:
                break
            for event in (_decode(row) for row in rows):
                if event["data"].get("released") is False:
                    rejected.append(event)
                    if len(rejected) >= limit:
                        break
            last_seq = rows[-1][0]
    finally:
        conn.close()
    return rejected[:limit]


def harvest_feedback(
    limit: int = 50,
    workspace_id: str | None = None,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Convert reviewer rejections into candidate evaluation cases for eval/golden.json."""
    rejections = list_rejections(limit=limit, workspace_id=workspace_id, path=path)
    candidates = []
    for item in rejections:
        data = item.get("data", {})
        q = data.get("question") or ""
        seq = item.get("seq", 0)
        cites = [source_file(c.split("] ")[-1]) for c in data.get("citations", []) if "]" in c] or data.get("citations", [])
        candidates.append({
            "id": f"harvested-feedback-{seq}",
            "split": "test",
            "workspace": item.get("workspace_id") or "default",
            "question": q,
            "expected_sources": cites,
            "expected_keywords": [],
            "reviewer_reason": data.get("reason", ""),
            "suite": "en",
            "lang": "en",
        })
    return candidates


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
    target = Path(anchor_path or settings.anchor_path)
    if not target.exists():
        return []
    with open(target, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


class AnchorCopyError(RuntimeError):
    """The primary anchor was written, but the second copy could not be brought up to date."""

    def __init__(self, entry: dict[str, Any], message: str):
        super().__init__(message)
        self.entry = entry


def _copy_target(anchor_path: str | None, copy_dir: str | None) -> Path | None:
    """The second anchor file, or None. AUDIT_ANCHOR_COPY_DIR applies to the live anchor file only: a backup's
    anchors, or any explicit file, are checked on their own unless a directory is passed."""
    chosen = copy_dir if copy_dir is not None else (settings.audit_anchor_copy_dir if anchor_path is None else "")
    if not chosen:
        return None
    return Path(chosen) / Path(anchor_path or settings.anchor_path).name


def _sync_copy(primary: list[dict[str, Any]], target: Path) -> None:
    """Append the anchors the copy lacks. Refuses to touch a copy that disagrees with the primary."""
    directory = target.parent
    if not directory.is_dir():
        if not directory.parent.is_dir():
            raise AnchorCopyError({}, f"{directory} is not reachable (is the drive or share mounted?)")
        directory.mkdir()
    try:
        existing = _read_anchors(str(target))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise AnchorCopyError({}, f"the copy {target} could not be read ({exc})") from exc
    if len(existing) > len(primary):
        raise AnchorCopyError({}, f"the copy {target} has more anchors ({len(existing)}) than the primary "
                                  f"({len(primary)}): the primary anchor file was truncated")
    for index, (mine, theirs) in enumerate(zip(primary, existing), 1):
        if mine != theirs:
            raise AnchorCopyError({}, f"the copy {target} differs from the primary at anchor {index}: "
                                      "one of them was edited, so nothing was appended")
    missing = primary[len(existing):]
    if not missing:
        return
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        for entry in missing:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def anchor(
    path: str | None = None, anchor_path: str | None = None, copy_dir: str | None = None
) -> dict[str, Any] | None:
    """Record the current head hash in the anchor file. Each anchor is signed and chained to the previous one.

    With AUDIT_ANCHOR_COPY_DIR (or `copy_dir`) the anchors are also written to a second location, which should be
    somewhere an attacker on this host cannot rewrite. The primary is written first; if the copy then fails,
    AnchorCopyError carries the new anchor and says why, and the command exits non-zero.
    """
    conn = _open(path)
    try:
        head = conn.execute("SELECT seq, record_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if head is None:
        return None
    target = Path(anchor_path or settings.anchor_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # The read-modify-append runs under an exclusive lock on the anchor file: two overlapping runs
    # (cron plus manual) would otherwise chain onto the same previous anchor and interleave lines.
    with open(target, "a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # platforms without fcntl keep the single-writer behaviour of the append itself
        try:
            anchors = _read_anchors(str(target))
            entry: dict[str, Any] = {
                "seq": head[0],
                "hash": head[1],
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "prev": anchors[-1]["mac"] if anchors else GENESIS_HASH,
            }
            entry["mac"] = anchor_mac(_canonical(entry))
            handle.seek(0, os.SEEK_END)
            handle.write((json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
    copy = _copy_target(anchor_path, copy_dir)
    if copy is not None:
        try:
            _sync_copy([*anchors, entry], copy)
        except AnchorCopyError as exc:
            raise AnchorCopyError(entry, str(exc)) from None
        except OSError as exc:
            raise AnchorCopyError(entry, f"the copy {copy} could not be written ({exc})") from exc
    return entry


def _check_anchor_file(conn: sqlite3.Connection, target: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Check one anchor file's signatures, chaining, and that the anchored records still exist unchanged."""
    problems: list[str] = []
    anchors: list[dict[str, Any]] = []
    if target.exists():
        with open(target, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"{label}anchor line {line_no}: invalid JSON (anchor file was edited)")
                    continue
                if not isinstance(parsed, dict):
                    problems.append(f"{label}anchor line {line_no}: entry is not a JSON object")
                    continue
                anchors.append(parsed)
    base_seq, base_hash = _base(conn)
    problems.extend(_base_problems(conn))
    prev = GENESIS_HASH
    for index, entry in enumerate(anchors, 1):
        body = {key: value for key, value in entry.items() if key != "mac"}
        if not hmac.compare_digest(anchor_mac(_canonical(body)), str(entry.get("mac", ""))):
            problems.append(f"{label}anchor {index}: signature does not match (anchor file was edited)")
        if entry.get("prev") != prev:
            problems.append(f"{label}anchor {index}: chaining to the previous anchor is broken")
        prev = str(entry.get("mac", ""))

        try:
            seq = int(entry["seq"])
            expected_hash = str(entry["hash"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{label}anchor {index}: missing or invalid seq/hash fields")
            continue
        row = conn.execute("SELECT record_hash FROM audit_events WHERE seq = ?", (seq,)).fetchone()
        if row is not None:
            if row[0] != expected_hash:
                problems.append(f"{label}anchor {index}: record {seq} changed after it was anchored")
        elif seq == base_seq:
            if base_hash != expected_hash:
                problems.append(f"{label}anchor {index}: retention base does not match the anchored hash")
        elif seq > base_seq:
            problems.append(f"{label}anchor {index}: anchored record {seq} is missing; the log was truncated")
    return anchors, problems


def verify_anchors(
    path: str | None = None, anchor_path: str | None = None, copy_dir: str | None = None
) -> tuple[bool, list[str], int]:
    """Check every anchor file on its own, then the primary against the second copy when one is configured.

    Anchors older than the retention base describe records that were deleted on purpose; they are
    accepted without a record to compare against. Returns (ok, problems, anchors in the primary file).

    The two files are compared by signature: a copy with more anchors than the primary means the primary was
    truncated, a different anchor at the same position means one was edited, and a copy with fewer is behind
    (`make audit-anchor` brings it up to date). Both files being intact and equal is what makes deleting or
    rewriting the primary on this host detectable.
    """
    target = Path(anchor_path or settings.anchor_path)
    copy = _copy_target(anchor_path, copy_dir)
    conn = _open(path)
    try:
        anchors, problems = _check_anchor_file(conn, target, "")
        if copy is not None:
            copy_anchors: list[dict[str, Any]] = []
            if not copy.parent.is_dir():
                problems.append(f"copy: {copy.parent} is not reachable (is the drive or share mounted?)")
            elif not copy.exists():
                if anchors:
                    problems.append(f"copy: {copy} does not exist; {len(anchors)} anchor(s) are not copied "
                                    "(run 'make audit-anchor')")
            else:
                copy_anchors, copy_problems = _check_anchor_file(conn, copy, "copy: ")
                problems.extend(copy_problems)
            if copy.exists():
                for index, (mine, theirs) in enumerate(zip(anchors, copy_anchors), 1):
                    if mine != theirs:
                        problems.append(f"copy: anchor {index} differs from the primary: one of them was edited")
                        break
                else:
                    if len(copy_anchors) > len(anchors):
                        problems.append(f"copy: has {len(copy_anchors)} anchor(s) but the primary has {len(anchors)}: "
                                        "the primary anchor file was truncated")
                    elif len(copy_anchors) < len(anchors):
                        problems.append(f"copy: is behind by {len(anchors) - len(copy_anchors)} anchor(s) "
                                        "(run 'make audit-anchor' to bring it up to date)")
    finally:
        conn.close()
    return not problems, problems, len(anchors)


# --- Retention --------------------------------------------------------------

def prune(retention_days: int | None = None, path: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Delete records older than the retention period. The deletion is recorded in the chain.

    The chain base moves to the hash of the last deleted record, so the remaining log still verifies.
    """
    retention_days = settings.retention_days if retention_days is None else retention_days
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
        conn.execute(
            "INSERT OR REPLACE INTO audit_meta (key, value) VALUES ('base_mac', ?)",
            (base_mac(json.dumps([str(last_seq), last_hash], separators=(",", ":"))),),
        )
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


def main(argv: list[str] | None = None, path: str | None = None) -> int:
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
    harvest = sub.add_parser("harvest", help="Extract reviewer rejections as candidate eval cases")
    harvest.add_argument("-n", type=int, default=50)
    harvest.add_argument("--workspace")
    harvest.add_argument("-o", "--output", help="Optional file path to write candidate cases to")
    export_p = sub.add_parser("export", help="Print every event, oldest first, as JSON lines (decrypted)")
    export_p.add_argument("-o", "--output", help="Optional file path to write JSONL export to")
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
        try:
            entry = anchor()
        except AnchorCopyError as exc:
            print(f"Anchored seq {exc.entry['seq']}: {exc.entry['hash']}")
            print(f"Error: the second anchor copy is NOT up to date: {exc}", file=sys.stderr)
            return 1
        print("Nothing to anchor: the log is empty." if entry is None else f"Anchored seq {entry['seq']}: {entry['hash']}")
        if entry is not None and settings.audit_anchor_copy_dir:
            print(f"Copied to {settings.audit_anchor_copy_dir}.")
        return 0

    if args.command == "head":
        print(head_hash())
        return 0

    if args.command == "feedback":
        rejections = list_rejections(limit=args.n, workspace_id=args.workspace, path=path)
        if not rejections:
            print("No rejected answers recorded.")
        for item in rejections:
            data = item["data"]
            print(f"{item['ts']} [{item['workspace_id']}] approver={data.get('approver') or '-'}")
            if data.get("question"):
                print(f"  question: {data['question']}")
            print(f"  reason: {data.get('reason') or '(none given)'}")
            print(f"  groundedness: {data.get('groundedness_ratio')}  degraded: {data.get('degraded') or 'no'}")
            print(f"  citations: {', '.join(data.get('citations', [])) or 'none'}")
        return 0

    if args.command == "harvest":
        candidates = harvest_feedback(limit=args.n, workspace_id=args.workspace, path=path)
        formatted = json.dumps(candidates, indent=2, ensure_ascii=False)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(formatted + "\n", encoding="utf-8")
            print(f"Harvested {len(candidates)} candidate eval case(s) to {out_path}")
        else:
            print(formatted)
        return 0

    if args.command == "export":
        events = export_events()
        record("audit_export", workspace_id="", session_id="", data={"actor": "cli", "records": len(events)})
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(f"Exported {len(events)} audit record(s) to {out_path}")
        else:
            for event in events:
                _print_event(event)
        return 0

    for event in reversed(list_events(limit=args.n, workspace_id=args.workspace)):
        _print_event(event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
