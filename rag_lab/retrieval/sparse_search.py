"""Safe, embedded SQLite FTS5 keyword indexer and searcher.

Uses native SQLite Full-Text Search (FTS5) BM25 scoring: no deserialization of stored objects
(CWE-502), zero extra dependencies, and a small memory footprint. Legacy `.pkl` indexes from older
versions are never loaded; re-index the workspace instead.

Retrieval is fail-closed. A missing clearance column or a database error raises SparseIndexError so
the caller can report the keyword index as unavailable, and a query without an explicit clearance
list sees public chunks only.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rag_lab.config import BM25_DIR

logger = logging.getLogger("sparse_search")


class SparseIndexError(RuntimeError):
    """The keyword index exists but cannot be searched safely (old schema, corruption, or a database error)."""


def get_fts_db_path(workspace_id: str, base_dir: str | None = None) -> str:
    """Return the filesystem path to the workspace's SQLite FTS5 database."""
    target_dir = Path(base_dir or BM25_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    return str(target_dir / f"{workspace_id}.db")


def index_sparse_chunks(
    workspace_id: str,
    chunk_ids: list[str],
    sources: list[str],
    chunks: list[str],
    clearances: list[str] | None = None,
    base_dir: str | None = None,
) -> None:
    """Index chunks into an isolated SQLite FTS5 virtual table for keyword retrieval.

    Supports document clearance tags (e.g. 'public', 'internal', 'confidential', 'secret').
    """
    db_path = get_fts_db_path(workspace_id, base_dir)
    if clearances is None:
        clearances = ["public"] * len(chunk_ids)

    try:
        with closing(sqlite3.connect(db_path)) as conn:
            # Recreate table with clearance column if old schema exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks';")
            if cursor.fetchone():
                info = conn.execute("PRAGMA table_info(fts_chunks);").fetchall()
                cols = [r[1] for r in info]
                if "clearance" not in cols:
                    conn.execute("DROP TABLE fts_chunks;")

            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
                "id UNINDEXED, "
                "source UNINDEXED, "
                "clearance UNINDEXED, "
                "content, "
                "tokenize='porter unicode61'"
                ");"
            )
            conn.execute("DELETE FROM fts_chunks;")
            rows = [
                (cid, src, clr, content)
                for cid, src, clr, content in zip(chunk_ids, sources, clearances, chunks)
            ]
            conn.executemany("INSERT INTO fts_chunks (id, source, clearance, content) VALUES (?, ?, ?, ?);", rows)
            conn.commit()
    except Exception as exc:
        logger.exception("Failed to write SQLite FTS5 index for %s: %s", workspace_id, exc)
        raise


def search_sparse_chunks(
    workspace_id: str,
    query: str,
    top_n: int = 5,
    allowed_clearances: list[str] | None = None,
    base_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Query the sparse index using SQLite FTS5 BM25 ranking, filtered by allowed clearance levels.

    `allowed_clearances=None` means public only; an empty list returns nothing. Returns [] when the
    workspace has no keyword index. Raises SparseIndexError when an index exists but cannot be
    searched under the clearance filter.
    """
    clearances = ["public"] if allowed_clearances is None else list(allowed_clearances)
    if not clearances:
        return []

    target_dir = base_dir or BM25_DIR
    db_path = os.path.join(target_dir, f"{workspace_id}.db")
    if not os.path.exists(db_path):
        return []

    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    match_query = " OR ".join(f'"{t}"' for t in tokens)
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(fts_chunks);").fetchall()]
            if "clearance" not in columns:
                raise SparseIndexError(
                    f"keyword index for '{workspace_id}' has no clearance column; re-index the workspace"
                )
            placeholders = ", ".join("?" for _ in clearances)
            rows = conn.execute(
                "SELECT id, source, content, clearance, bm25(fts_chunks) AS score "
                "FROM fts_chunks "
                f"WHERE fts_chunks MATCH ? AND clearance IN ({placeholders}) "
                "ORDER BY score ASC LIMIT ?;",
                [match_query, *clearances, top_n],
            ).fetchall()
    except SparseIndexError:
        raise
    except sqlite3.Error as exc:
        logger.warning("SQLite FTS5 query error on workspace %s: %s", workspace_id, exc)
        raise SparseIndexError(f"keyword index for '{workspace_id}' could not be searched: {exc}") from exc
    return [
        {"id": row[0], "content": row[2], "citation": f"[{row[0]}] {row[1]}", "clearance": row[3]}
        for row in rows
    ]


def delete_sparse_index(workspace_id: str, base_dir: str | None = None) -> None:
    """Safely delete all sparse index files (.db, .db-wal, .db-shm, and leftover legacy .pkl) for a workspace."""
    target_dir = Path(base_dir or BM25_DIR)
    for ext in [".db", ".db-wal", ".db-shm", ".pkl", ".db.tmp", ".pkl.tmp"]:
        fpath = target_dir / f"{workspace_id}{ext}"
        if fpath.exists():
            try:
                fpath.unlink()
            except OSError as exc:
                logger.debug("Could not remove %s: %s", fpath, exc)


def indexed_clearances(workspace_id: str, base_dir: str | None = None) -> set[str]:
    """Clearance levels present among a workspace's indexed chunks; empty when it is not indexed."""
    db_path = get_fts_db_path(workspace_id, base_dir)
    if not os.path.exists(db_path):
        return set()
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute("SELECT DISTINCT clearance FROM fts_chunks;").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[0]).strip().lower() for row in rows if row[0]}


def has_sparse_index(workspace_id: str, base_dir: str | None = None) -> bool:
    """Return True if an FTS5 index exists for the workspace."""
    return (Path(base_dir or BM25_DIR) / f"{workspace_id}.db").exists()
