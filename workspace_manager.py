#!/usr/bin/env python3
import os
import re
import shutil
from typing import List, Dict, Any
import chromadb
from psycopg_pool import ConnectionPool

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
BM25_DIR = os.getenv("BM25_DIR", "./bm25_indices")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")
DB_URI = os.getenv(
    "DB_URI",
    "postgresql://rag_user:rag_password@localhost:5432/rag_checkpoint_db"
)


def sanitize_workspace_id(raw_name: str) -> str:
    """Sanitizes user input into a safe filesystem/collection slug.
    Collapses spaces, dashes, and symbols into single underscores.
    """
    cleaned = raw_name.strip().lower()
    # Replace any sequence of non-alphanumeric characters with a single underscore
    cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
    return cleaned.strip('_')


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
    except Exception:
        pass

    # 2. Discover serialized BM25 indices
    if os.path.exists(BM25_DIR):
        for f in os.listdir(BM25_DIR):
            if f.endswith(".pkl"):
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
    except Exception:
        pass

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
    except Exception:
        pass

    # 2. Delete BM25 pickle
    bm25_file = os.path.join(BM25_DIR, f"{ws_id}.pkl")
    if os.path.exists(bm25_file):
        try:
            os.remove(bm25_file)
        except OSError:
            pass

    # 3. Delete uploaded source files
    upload_path = os.path.join(UPLOADS_DIR, ws_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path, ignore_errors=True)

    # 4. Clean up Postgres checkpoint records for this workspace prefix
    try:
        with ConnectionPool(conninfo=DB_URI, max_size=2, open=True, kwargs={"autocommit": True}) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoints WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id LIKE %s;", (f"ui-{ws_id}-%",))
    except Exception:
        # If Postgres is temporarily offline, don't break file deletion
        pass

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
    except Exception:
        pass

    # BM25 presence
    bm25_file = os.path.join(BM25_DIR, f"{ws_id}.pkl")
    bm25_indexed = os.path.exists(bm25_file)

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
