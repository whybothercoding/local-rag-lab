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
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from audit import head_hash, verify_anchors, verify_chain
from checkpointer import get_sqlite_path, is_postgres_uri
from config import ACCESS_POLICY_PATH, AUDIT_DB_PATH, ANCHOR_PATH, BACKUP_DIR, DB_URI

MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_copy(source: Path, destination: Path) -> None:
    """Copy a live SQLite database with its backup API, so the copy is consistent even under writes."""
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)


def create(destination: Path | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = destination or Path(BACKUP_DIR) / f"backup-{stamp}"
    target.mkdir(parents=True, exist_ok=False)

    files: dict[str, str] = {}
    _sqlite_copy(Path(AUDIT_DB_PATH), target / "audit.db")
    files["audit.db"] = _sha256(target / "audit.db")

    if not is_postgres_uri(DB_URI):
        checkpoints = Path(get_sqlite_path(DB_URI))
        if checkpoints.exists():
            _sqlite_copy(checkpoints, target / "checkpoints.db")
            files["checkpoints.db"] = _sha256(target / "checkpoints.db")

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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif _sha256(path) != expected:
            problems.append(f"{name} does not match its checksum")

    audit_copy = directory / "audit.db"
    if audit_copy.exists():
        chain_ok, bad_seq, _count = verify_chain(str(audit_copy))
        if not chain_ok:
            problems.append(f"audit chain broken at seq {bad_seq}")
        anchors_ok, anchor_problems, _anchors = verify_anchors(
            str(audit_copy), str(directory / "audit_anchors.jsonl")
        )
        if not anchors_ok:
            problems.extend(f"anchors: {problem}" for problem in anchor_problems)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and verify governance backups.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Create a backup in backups/")
    check = sub.add_parser("verify", help="Verify a backup directory")
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "create":
        target = create()
        print(f"Backup written to {target}. Keys were not included; store them separately.")
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
