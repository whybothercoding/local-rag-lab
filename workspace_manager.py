#!/usr/bin/env python3
import logging
import os
import re
import shutil
from typing import List, Dict, Any
import chromadb
from psycopg_pool import ConnectionPool

from config import (
    BM25_DIR as _BM25_DIR,
    CHROMA_PATH as _CHROMA_PATH,
    DB_URI as _DB_URI,
    UPLOADS_DIR as _UPLOADS_DIR,
)
from checkpointer import is_postgres_uri, get_sqlite_path
from sparse_search import delete_sparse_index, has_sparse_index

# Re-exported so tests and ingest can redirect storage without editing config.
CHROMA_PATH = _CHROMA_PATH
BM25_DIR = _BM25_DIR
UPLOADS_DIR = _UPLOADS_DIR
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

    return sorted(list(workspaces))


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

    # 3. Delete uploaded source files
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

    # Source files in staging
    upload_path = os.path.join(UPLOADS_DIR, ws_id)
    if os.path.exists(upload_path):
        file_count = len([f for f in os.listdir(upload_path) if not f.startswith(".")])

    return {
        "workspace_id": ws_id,
        "files": file_count,
        "chunks": chunk_count,
        "bm25_indexed": bm25_indexed
    }
