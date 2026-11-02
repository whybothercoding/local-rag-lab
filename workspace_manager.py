#!/usr/bin/env python3
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from psycopg_pool import ConnectionPool

from config import (
    BM25_DIR as _BM25_DIR,
    CHROMA_PATH as _CHROMA_PATH,
    DB_URI as _DB_URI,
    UPLOADS_DIR as _UPLOADS_DIR,
    KNOWLEDGE_BASE_DIR as _KNOWLEDGE_BASE_DIR,
)
from checkpointer import is_postgres_uri, get_sqlite_path
from parsers import SUPPORTED_EXTENSIONS
from sparse_search import delete_sparse_index, has_sparse_index

# Re-exported so tests and ingest can redirect storage without editing config.
CHROMA_PATH = _CHROMA_PATH
BM25_DIR = _BM25_DIR
UPLOADS_DIR = _UPLOADS_DIR
KNOWLEDGE_BASE_DIR = _KNOWLEDGE_BASE_DIR
DB_URI = _DB_URI

logger = logging.getLogger("workspace_manager")

# `ws_<id>__build` must stay inside Chroma's 63-character collection name limit.
_MAX_WORKSPACE_ID = 48


def sanitize_workspace_id(raw_name: str) -> str:
    """Sanitizes user input into a safe filesystem/collection slug.
    Collapses spaces, dashes, and symbols into single underscores.
    """
    cleaned = raw_name.strip().lower()
    # Replace any sequence of non-alphanumeric characters with a single underscore
    cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
    return cleaned.strip('_')[:_MAX_WORKSPACE_ID].strip('_')


def safe_upload_name(raw_name: str) -> str:
    """Reduce an uploaded filename to a plain name with a supported extension.

    Directory parts, hidden names, control characters, and unsupported types are refused.
    """
    name = os.path.basename(raw_name.replace("\\", "/")).strip()
    if not name or name in {".", ".."} or name.startswith("."):
        raise ValueError("Invalid file name.")
    if any(ord(ch) < 32 or ch in '<>:"|?*' for ch in name):
        raise ValueError(f"File name '{name!r}' contains unsupported characters.")
    if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type for '{name}'.")
    return name


def staged_path(ws_id: str, raw_name: str) -> Path:
    """Where an upload for a workspace is staged. Always inside that workspace's upload folder."""
    base = (Path(UPLOADS_DIR) / ws_id).resolve()
    target = (base / safe_upload_name(raw_name)).resolve()
    if target.parent != base:
        raise ValueError("Invalid file name.")
    return target


def ensure_base_dirs():
    """Ensure all required root directories exist on disk."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(BM25_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def list_workspaces() -> List[str]:
    """Discovers all workspaces across ChromaDB collections, BM25 indices, and upload dirs."""
    ensure_base_dirs()
    workspaces = set()

    # 1. Discover collections from ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        for col in client.list_collections():
            name = getattr(col, "name", str(col))
            if name.startswith("ws_"):
                workspaces.add(name[3:])
    except Exception as exc:
        logger.warning("Could not list Chroma collections: %s", exc)

    # 2. Discover serialized sparse indices (SQLite FTS5 and legacy PKL)
    if os.path.exists(BM25_DIR):
        for f in os.listdir(BM25_DIR):
            if f.endswith(".db") and not f.endswith(("-wal", "-shm")):
                workspaces.add(f[:-3])
            elif f.endswith(".pkl"):
                workspaces.add(f[:-4])

    # 3. Discover staged upload folders
    if os.path.exists(UPLOADS_DIR):
        for entry in os.listdir(UPLOADS_DIR):
            full_path = os.path.join(UPLOADS_DIR, entry)
            if os.path.isdir(full_path) and not entry.startswith("."):
                workspaces.add(entry)

    # 4. Discover knowledge_base folders (same rules as sync)
    for ws_id, _ in discover_knowledge_base_workspaces():
        workspaces.add(ws_id)

    return sorted(list(workspaces))


def discover_knowledge_base_workspaces(base_dir: str | None = None) -> list[Tuple[str, str]]:
    """Return (workspace_id, directory) for each plain folder under knowledge_base/.

    Symlinked folders are skipped so a link cannot pull content from outside the tree.
    When two folder names sanitize to the same ID, the first one wins.
    """
    kb_path = Path(base_dir or KNOWLEDGE_BASE_DIR)
    if not kb_path.is_dir():
        return []

    discovered: list[Tuple[str, str]] = []
    seen: set[str] = set()
    for entry in sorted(kb_path.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        ws_id = sanitize_workspace_id(entry.name)
        if not ws_id:
            continue
        if ws_id in seen:
            logger.warning("Skipping '%s': workspace id '%s' is already used by another folder.", entry.name, ws_id)
            continue
        seen.add(ws_id)
        discovered.append((ws_id, str(entry)))
    return discovered


def sync_all_workspaces(
    base_dir: str | None = None,
    force: bool = False,
    actor: str = "system",
) -> Dict[str, Any]:
    """Scan knowledge_base/ directory and automatically index all discovered workspaces.

    Idempotent: Uses content-addressable SHA-256 manifests to skip unchanged files.
    """
    from ingest import ingest_workspace
    discovered = discover_knowledge_base_workspaces(base_dir)
    results = {}
    for ws_id, dir_path in discovered:
        results[ws_id] = ingest_workspace(ws_id, dir_path, force=force, actor=actor)
    return results


def create_workspace(raw_name: str) -> str:
    """Initializes a new empty workspace staging area and vector collection.
    Returns the sanitized workspace ID, or empty string on invalid name.
    """
    ensure_base_dirs()
    ws_id = sanitize_workspace_id(raw_name)
    if not ws_id:
        return ""

    # Ensure staging folder exists
    os.makedirs(os.path.join(UPLOADS_DIR, ws_id), exist_ok=True)

    # Initialize empty collection in ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.get_or_create_collection(name=f"ws_{ws_id}")
    except Exception as exc:
        logger.warning("Could not create Chroma collection for %s: %s", ws_id, exc)

    return ws_id


def delete_workspace(raw_name: str) -> bool:
    """Completely purges a workspace across:
    1. ChromaDB vector collection
    2. BM25 pickle index
    3. Uploaded source files
    4. LangGraph PostgreSQL checkpoint history for matching threads
    """
    ws_id = sanitize_workspace_id(raw_name)
    if not ws_id:
        return False

    # 1. Delete Chroma collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(name=f"ws_{ws_id}")
    except Exception as exc:
        logger.warning("Could not delete Chroma collection for %s: %s", ws_id, exc)

    # 2. Delete sparse index (SQLite FTS5 and legacy PKL)
    delete_sparse_index(ws_id, base_dir=BM25_DIR)

    # 3. Delete manifest file if present
    manifest_file = os.path.join(BM25_DIR, f"{ws_id}_manifest.json")
    if os.path.exists(manifest_file):
        try:
            os.remove(manifest_file)
        except OSError:
            pass

    # 4. Delete uploaded source files
    upload_path = os.path.join(UPLOADS_DIR, ws_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path, ignore_errors=True)

    # 4. Clean up checkpoint records for this workspace prefix
    if is_postgres_uri(DB_URI):
        try:
            with ConnectionPool(conninfo=DB_URI, max_size=2, open=True, kwargs={"autocommit": True}) as pool:
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM checkpoints WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
                        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
                        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
        except Exception as exc:
            logger.warning("Could not delete checkpoints for %s: %s", ws_id, exc)
    else:
        try:
            sqlite_db = get_sqlite_path(DB_URI)
            if os.path.exists(sqlite_db):
                import sqlite3
                with sqlite3.connect(sqlite_db) as conn:
                    conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE ?;", (f"ui-{ws_id}-%",))
                    conn.execute("DELETE FROM writes WHERE thread_id LIKE ?;", (f"ui-{ws_id}-%",))
        except Exception as exc:
            logger.warning("Could not delete SQLite checkpoints for %s: %s", ws_id, exc)

    return True


def get_workspace_stats(raw_name: str) -> Dict[str, Any]:
    """Retrieves document count, chunk count, and indexed status."""
    ws_id = sanitize_workspace_id(raw_name)
    chunk_count = 0
    file_count = 0
    bm25_indexed = False

    # Chunks from Chroma
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(name=f"ws_{ws_id}")
        chunk_count = col.count()
    except Exception as exc:
        logger.warning("Could not read chunk count for %s: %s", ws_id, exc)

    # Sparse index presence
    bm25_indexed = has_sparse_index(ws_id, base_dir=BM25_DIR)

    # 1. Manifest count (tracks all indexed files across knowledge_base and uploads)
    manifest_file = os.path.join(BM25_DIR, f"{ws_id}_manifest.json")
    if os.path.exists(manifest_file):
        try:
            import json
            with open(manifest_file, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
                file_count = len(manifest.get("files", {}))
        except Exception:
            pass

    # 2. Fallback / union with unindexed files on disk
    seen_files = set()
    upload_path = os.path.join(UPLOADS_DIR, ws_id)
    if os.path.exists(upload_path):
        for f in os.listdir(upload_path):
            if not f.startswith(".") and os.path.isfile(os.path.join(upload_path, f)):
                seen_files.add(f)

    for disc_id, kb_dir in discover_knowledge_base_workspaces():
        if disc_id == ws_id and os.path.exists(kb_dir):
            for f in os.listdir(kb_dir):
                if not f.startswith(".") and os.path.isfile(os.path.join(kb_dir, f)):
                    seen_files.add(f)

    file_count = max(file_count, len(seen_files))

    return {
        "workspace_id": ws_id,
        "files": file_count,
        "chunks": chunk_count,
        "bm25_indexed": bm25_indexed
    }
