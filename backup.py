#!/usr/bin/env python3
"""Consistent, checksummed backups of the governance data.

Backed up: the audit log, the audit anchors, the checkpoint database, and the access policy.

Not backed up, on purpose:
- Keys (data.key, access.key). Keep them in a separate custody from the backups. A backup without
  its keys cannot be read, which is what separate custody is for. Losing the data key makes the
  audit log and checkpoints unreadable, so record where it is kept.
- Vector and keyword indexes. They are rebuilt from the source documents with `make sync`.

Usage:
  python backup.py create            # writes backups/backup-<timestamp>/
  python backup.py verify <dir>      # checks checksums, the audit chain, and the anchors in a backup
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from audit import head_hash, verify_anchors, verify_chain
from checkpointer import get_sqlite_path, is_postgres_uri
from config import ACCESS_POLICY_PATH, AUDIT_DB_PATH, ANCHOR_PATH, BACKUP_DIR, CACHE_DB_PATH, DB_URI

MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_copy(source: Path, destination: Path) -> None:
    """Copy a live SQLite database with its backup API, so the copy is consistent even under writes."""
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)


def create(destination: Path | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = destination or Path(BACKUP_DIR) / f"backup-{stamp}"
    target.mkdir(parents=True, exist_ok=False)

    files: dict[str, str] = {}
    # Ensure audit database and schema exist even on a fresh deployment with 0 events
    verify_chain(str(AUDIT_DB_PATH))
    _sqlite_copy(Path(AUDIT_DB_PATH), target / "audit.db")
    files["audit.db"] = _sha256(target / "audit.db")

    if not is_postgres_uri(DB_URI):
        checkpoints = Path(get_sqlite_path(DB_URI))
        if checkpoints.exists():
            _sqlite_copy(checkpoints, target / "checkpoints.db")
            files["checkpoints.db"] = _sha256(target / "checkpoints.db")

    cache_db = Path(CACHE_DB_PATH)
    if cache_db.exists():
        _sqlite_copy(cache_db, target / "cache.db")
        files["cache.db"] = _sha256(target / "cache.db")

    for source, name in ((Path(ANCHOR_PATH), "audit_anchors.jsonl"), (Path(ACCESS_POLICY_PATH), "access_policy.json")):
        if source.exists():
            shutil.copy2(source, target / name)
            files[name] = _sha256(target / name)

    chain_ok, _bad, records = verify_chain(str(target / "audit.db"))
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        "audit_records": records,
        "audit_head": head_hash(str(target / "audit.db")),
        "audit_chain_ok": chain_ok,
        "checkpoints_backend": "postgres (external pg_dump required)" if is_postgres_uri(DB_URI) else "sqlite",
        "excluded": ["data.key", "access.key"],
    }
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def verify(directory: Path) -> list[str]:
    """Return a list of problems. An empty list means the backup is intact."""
    problems: list[str] = []
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return [f"no {MANIFEST} in {directory}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"corrupted {MANIFEST} in {directory}"]
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return [f"malformed {MANIFEST} in {directory}"]

    if "audit.db" not in manifest["files"]:
        problems.append("audit.db is missing from manifest")

    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif _sha256(path) != expected:
            problems.append(f"{name} does not match its checksum")

    audit_copy = directory / "audit.db"
    if audit_copy.exists():
        chain_ok, bad_seq, count = verify_chain(str(audit_copy))
        if not chain_ok:
            problems.append(f"audit chain broken at seq {bad_seq}")
        if "audit_records" in manifest and manifest["audit_records"] != count:
            problems.append("audit record count does not match manifest")
        if "audit_head" in manifest and manifest["audit_head"] != head_hash(str(audit_copy)):
            problems.append("audit head hash does not match manifest")
        anchors_ok, anchor_problems, _anchors = verify_anchors(
            str(audit_copy), str(directory / "audit_anchors.jsonl")
        )
        if not anchors_ok:
            problems.extend(f"anchors: {problem}" for problem in anchor_problems)
    return problems


def list_backups(base_dir: Path | None = None) -> list[dict]:
    """Return metadata for each backup directory under backups/, newest first."""
    root = base_dir or Path(BACKUP_DIR)
    if not root.is_dir():
        return []
    entries = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST
        info = {"path": str(child), "name": child.name, "created": "-", "audit_records": 0}
        if manifest_path.is_file():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                info["created"] = str(data.get("created", "-"))
                info["audit_records"] = int(data.get("audit_records", 0))
            except Exception:
                pass
        entries.append(info)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and verify governance backups.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Create a backup in backups/")
    sub.add_parser("list", help="List existing backups in backups/")
    check = sub.add_parser("verify", help="Verify a backup directory")
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "create":
        target = create()
        print(f"Backup written to {target}. Keys were not included; store them separately.")
        return 0

    if args.command == "list":
        backups = list_backups()
        if not backups:
            print("No backups found.")
            return 0
        for b in backups:
            print(f"{b['path']}  (created={b['created']}, audit_records={b['audit_records']})")
        return 0

    problems = verify(args.directory)
    if problems:
        for problem in problems:
            print(f"FAILED: {problem}", file=sys.stderr)
        return 1
    print(f"Backup verified: {args.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
