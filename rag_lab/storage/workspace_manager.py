#!/usr/bin/env python3
import logging
import os
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from psycopg_pool import ConnectionPool

from rag_lab.config import settings
from rag_lab.storage.checkpointer import is_postgres_uri, get_sqlite_path
from rag_lab.ingestion.parsers import SUPPORTED_EXTENSIONS
from rag_lab.retrieval.sparse_search import delete_sparse_index, has_sparse_index


logger = logging.getLogger("workspace_manager")

# `ws_<id>__build` must stay inside Chroma's 63-character collection name limit.
_MAX_WORKSPACE_ID = 48
_TEMP_SUFFIXES = ("__build", "__old")


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
    base = (Path(settings.uploads_dir) / ws_id).resolve()
    target = (base / safe_upload_name(raw_name)).resolve()
    if target.parent != base:
        raise ValueError("Invalid file name.")
    return target


def ensure_base_dirs():
    """Ensure all required root directories exist on disk."""
    os.makedirs(settings.chroma_path, exist_ok=True)
    os.makedirs(settings.bm25_dir, exist_ok=True)
    os.makedirs(settings.uploads_dir, exist_ok=True)


def list_workspaces() -> List[str]:
    """Discovers all workspaces across ChromaDB collections, BM25 indices, and upload dirs."""
    ensure_base_dirs()
    workspaces = set()

    # 1. Discover collections from ChromaDB (not the temporary __build/__old copies used while re-indexing)
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        for col in client.list_collections():
            name = getattr(col, "name", str(col))
            if name.startswith("ws_") and not name.endswith(_TEMP_SUFFIXES):
                workspaces.add(name[3:])
    except Exception as exc:
        logger.warning("Could not list Chroma collections: %s", exc)

    # 2. Discover SQLite FTS5 keyword indices
    if os.path.exists(settings.bm25_dir):
        for f in os.listdir(settings.bm25_dir):
            if f.endswith(".db"):
                workspaces.add(f[:-3])

    # 3. Discover staged upload folders
    if os.path.exists(settings.uploads_dir):
        for entry in os.listdir(settings.uploads_dir):
            full_path = os.path.join(settings.uploads_dir, entry)
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
    kb_path = Path(base_dir or settings.knowledge_base_dir)
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
    from rag_lab.ingestion.ingest import ingest_workspace
    discovered = discover_knowledge_base_workspaces(base_dir)
    results = {}
    for ws_id, dir_path in discovered:
        results[ws_id] = ingest_workspace(ws_id, dir_path, force=force, actor=actor)
    return results


def _audit_workspace(event: str, ws_id: str, actor: str) -> None:
    from rag_lab.security.audit import record

    record(event, workspace_id=ws_id, session_id="", data={"actor": actor})


def create_workspace(raw_name: str, actor: str | None = None) -> str:
    """Initializes a new empty workspace staging area and vector collection.

    Returns the sanitized workspace ID, or empty string on invalid name. With `actor`, that account
    must be an administrator; the change is recorded in the audit log.
    """
    from rag_lab.security.access import require_admin

    who = require_admin(actor) if actor is not None else "system"
    ensure_base_dirs()
    ws_id = sanitize_workspace_id(raw_name)
    if not ws_id:
        return ""

    # Ensure staging folder exists
    os.makedirs(os.path.join(settings.uploads_dir, ws_id), exist_ok=True)

    # Initialize empty collection in ChromaDB
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        client.get_or_create_collection(name=f"ws_{ws_id}", metadata={"hnsw:space": "cosine"})
    except Exception as exc:
        logger.warning("Could not create Chroma collection for %s: %s", ws_id, exc)

    _audit_workspace("workspace_created", ws_id, who)
    return ws_id


def delete_workspace(raw_name: str, actor: str | None = None) -> bool:
    """Completely purges a workspace across:
    1. ChromaDB vector collection (and any leftover re-index copies)
    2. SQLite FTS5 keyword index
    3. Cached answers and the index manifest
    4. Uploaded source files
    5. LangGraph checkpoint history for matching threads
    6. Per-account grants for the workspace

    With `actor`, that account must be an administrator. The deletion is recorded in the audit log.
    """
    from rag_lab.security.access import remove_workspace_grants, require_admin

    who = require_admin(actor) if actor is not None else "system"
    ws_id = sanitize_workspace_id(raw_name)
    if not ws_id:
        return False

    # 1. Delete Chroma collections
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        for name in (f"ws_{ws_id}", f"ws_{ws_id}__build", f"ws_{ws_id}__old"):
            try:
                client.delete_collection(name=name)
            except Exception as exc:
                logger.debug("Collection %s not deleted: %s", name, exc)
    except Exception as exc:
        logger.warning("Could not delete Chroma collection for %s: %s", ws_id, exc)

    # 2. Delete sparse index (SQLite FTS5 and any leftover legacy PKL)
    delete_sparse_index(ws_id, base_dir=settings.bm25_dir)

    # 3. Delete semantic cache entries for this workspace
    try:
        from rag_lab.retrieval.semantic_cache import clear_workspace_cache
        clear_workspace_cache(ws_id)
    except Exception as exc:
        logger.debug("Could not clear cache for %s: %s", ws_id, exc)

    # 4. Delete manifest file if present
    manifest_file = os.path.join(settings.bm25_dir, f"{ws_id}_manifest.json")
    if os.path.exists(manifest_file):
        try:
            os.remove(manifest_file)
        except OSError:
            pass

    # 5. Delete uploaded source files
    upload_path = os.path.join(settings.uploads_dir, ws_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path, ignore_errors=True)

    # 6. Clean up checkpoint records for this workspace prefix
    escaped_ws = ws_id.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
    like_pattern = f"ui-{escaped_ws}-%"
    if is_postgres_uri(settings.db_uri):
        try:
            with ConnectionPool(conninfo=settings.db_uri, max_size=2, open=True, kwargs={"autocommit": True}) as pool:
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM checkpoints WHERE thread_id LIKE %s ESCAPE '\\';", (like_pattern,))
                        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id LIKE %s ESCAPE '\\';", (like_pattern,))
                        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id LIKE %s ESCAPE '\\';", (like_pattern,))
        except Exception as exc:
            logger.warning("Could not delete checkpoints for %s: %s", ws_id, exc)
    else:
        try:
            sqlite_db = get_sqlite_path(settings.db_uri)
            if os.path.exists(sqlite_db):
                with closing(sqlite3.connect(sqlite_db)) as conn:
                    tables = {
                        row[0]
                        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                    }
                    if "checkpoints" in tables:
                        conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE ? ESCAPE '\\';", (like_pattern,))
                    if "writes" in tables:
                        conn.execute("DELETE FROM writes WHERE thread_id LIKE ? ESCAPE '\\';", (like_pattern,))
                    conn.commit()
        except Exception as exc:
            logger.warning("Could not delete SQLite checkpoints for %s: %s", ws_id, exc)

    # 7. Remove stale grants for this workspace from the access policy
    try:
        remove_workspace_grants(ws_id)
    except Exception as exc:
        logger.warning("Could not prune grants for %s: %s", ws_id, exc)

    _audit_workspace("workspace_deleted", ws_id, who)
    return True


def get_workspace_stats(raw_name: str) -> Dict[str, Any]:
    """Retrieves document count, chunk count, and indexed status."""
    ws_id = sanitize_workspace_id(raw_name)
    chunk_count = 0
    file_count = 0
    bm25_indexed = False
    embedding = None

    # Chunks from Chroma
    try:
        from chromadb.errors import NotFoundError
        client = chromadb.PersistentClient(path=settings.chroma_path)
        col = client.get_collection(name=f"ws_{ws_id}")
        chunk_count = col.count()
    except NotFoundError:
        chunk_count = 0
    except Exception as exc:
        logger.warning("Could not read chunk count for %s: %s", ws_id, exc)

    # Sparse index presence
    bm25_indexed = has_sparse_index(ws_id, base_dir=settings.bm25_dir)

    # 1. Manifest count (tracks all indexed files across knowledge_base and uploads)
    manifest_file = os.path.join(settings.bm25_dir, f"{ws_id}_manifest.json")
    if os.path.exists(manifest_file):
        try:
            import json
            with open(manifest_file, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
                file_count = len(manifest.get("files", {}))
                embedding = manifest.get("embedding")
        except Exception:
            pass

    # 2. Fallback / union with unindexed files on disk
    seen_files = set()
    roots = [Path(settings.uploads_dir) / ws_id]
    for disc_id, kb_dir in discover_knowledge_base_workspaces():
        if disc_id == ws_id:
            roots.append(Path(kb_dir))

    for root in roots:
        if root.is_dir():
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and not path.name.startswith(".")
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    seen_files.add(str(path.relative_to(root)))

    file_count = max(file_count, len(seen_files))

    return {
        "workspace_id": ws_id,
        "files": file_count,
        "chunks": chunk_count,
        "bm25_indexed": bm25_indexed,
        "embedding": embedding,
    }


def print_status() -> int:
    """Print an end-to-end operational readiness and health summary."""
    from rag_lab.security.access import PolicyIntegrityError, describe_review_mode, list_users, review_mode_report
    from rag_lab.ops.approve import list_pending_threads
    from rag_lab.pipeline.graph import workflow
    from rag_lab.security.audit import verify_anchors, verify_chain
    from rag_lab.storage.checkpointer import get_checkpointer
    from rag_lab.config import settings
    from rag_lab.security.context_sanitizer import RUST_ACCEL
    from rag_lab.retrieval.embedding_provider import EmbeddingUnavailableError, embedding_fingerprint
    from rag_lab.pipeline.llm_provider import ollama_models
    from rag_lab.security.model_integrity import ModelIntegrityError, verify_model

    ok = True
    print("=== Local Enterprise RAG Lab: System Status ===")

    # 1. Embedding provider and pinned ONNX models. Only the models this configuration loads are required.
    runtime_fingerprint: str | None = None
    provider = ""
    try:
        runtime_fingerprint = embedding_fingerprint()
        provider = runtime_fingerprint.split(":", 1)[0]
        print(f"[OK] Model profile       : {settings.model_profile} (reranker {settings.reranker_model}, "
              f"relevance gate: {settings.relevance_gate})")
        print(f"[OK] Embedding provider  : {runtime_fingerprint}")
    except EmbeddingUnavailableError as exc:
        print(f"[!!] Embedding provider  : {exc}")
        ok = False
    required = ([settings.fastembed_model] if provider == "fastembed" else []) + ([settings.reranker_model] if settings.enable_reranker else [])
    if settings.groundedness_method == "nli":
        required.append(settings.nli_model)
    for model_name in required:
        try:
            verify_model(model_name, settings.models_dir)
            print(f"[OK] Model verified      : {model_name}")
        except ModelIntegrityError as exc:
            print(f"[!!] Model unverified    : {model_name} ({exc})")
            ok = False

    # 2. Rust accelerator & Ollama. Ollama is optional unless a setting depends on it.
    accel_state = "active (native PyO3)" if RUST_ACCEL else "fallback (pure Python; run 'make rust-build' to compile)"
    print(f"[OK] Context sanitizer   : {accel_state}")
    pulled = ollama_models(settings.ollama_base_url)
    chat_provider = settings.llm_provider.lower().strip()
    if pulled is None:
        if chat_provider == "ollama" or provider == "ollama":
            print(f"[!!] Ollama              : offline at {settings.ollama_base_url}, but the configuration requires it")
            ok = False
        else:
            print(f"[OK] LLM backend         : Ollama offline at {settings.ollama_base_url} (using deterministic in-process fallback)")
    else:
        if settings.chat_model in pulled:
            print(f"[OK] LLM backend         : reachable ({settings.chat_model} at {settings.ollama_base_url})")
        elif chat_provider in ("ollama", "auto"):
            print(f"[!!] LLM backend         : Ollama is up but {settings.chat_model} is not pulled (run 'make ollama-pull')")
            ok = False
        else:
            print(f"[--] LLM backend         : {chat_provider}; {settings.chat_model} is not pulled into Ollama")
        if provider == "ollama":
            if settings.embed_model in pulled:
                print(f"[OK] Ollama embeddings   : {settings.embed_model} pulled")
            else:
                print(f"[!!] Ollama embeddings   : {settings.embed_model} is not pulled (run 'make ollama-pull')")
                ok = False

    # 3. Accounts & RBAC policy
    try:
        users = list_users()
        admins = [u["username"] for u in users if u["admin"]]
        print(f"[OK] RBAC accounts       : {len(users)} user(s), {len(admins)} admin(s)")
        # Advisory: a release-mode gap is a governance choice to review, not a broken installation.
        report = review_mode_report(list_workspaces())
        for line in describe_review_mode(report):
            print(f"[OK] {line}")
        for warning in report.warnings:
            print(f"[--] Review coverage     : {warning}")
    except PolicyIntegrityError as exc:
        print(f"[!!] RBAC policy error   : {exc}")
        ok = False

    # 4. Workspaces
    workspaces = list_workspaces()
    if not workspaces:
        print("[--] Workspaces          : none (create with 'make create-workspace WS=<name>' or 'make sync ADMIN=<admin>')")
    else:
        print(f"[OK] Workspaces ({len(workspaces)})      :")
        for ws in workspaces:
            st = get_workspace_stats(ws)
            fts = "ready" if st["bm25_indexed"] else "not indexed"
            print(f"     - {ws:<18} files={st['files']}  chunks={st['chunks']}  fts5={fts}")
            if st["embedding"] and runtime_fingerprint and st["embedding"] != runtime_fingerprint:
                print(
                    f"[!!]   {ws}: indexed with {st['embedding']} but the runtime uses {runtime_fingerprint}; "
                    f"dense retrieval is disabled until 'make sync WS={ws} FORCE=1'"
                )
                ok = False

    # 5. Pending HITL reviews
    try:
        with get_checkpointer() as saver:
            app = workflow.compile(checkpointer=saver)
            pending = list_pending_threads(app)
        print(f"[OK] Pending HITL reviews: {len(pending)} thread(s)")
    except Exception as exc:
        print(f"[!!] Checkpointer error  : {exc}")
        ok = False

    # 6. Audit chain & anchors
    from rag_lab.config import settings
    if not Path(settings.audit_db_path).exists():
        print("[OK] Audit log integrity : 0 record(s) verified (not initialized yet)")
    else:
        chain_ok, bad_seq, count = verify_chain()
        anchors_ok, problems, anchor_count = verify_anchors()
        if chain_ok and anchors_ok:
            print(f"[OK] Audit log integrity : {count} record(s) verified, {anchor_count} anchor(s) intact")
        else:
            if not chain_ok:
                print(f"[!!] Audit chain broken  : seq {bad_seq}")
            for problem in problems:
                print(f"[!!] Audit anchor error  : {problem}")
            ok = False

    # 7. Semantic cache
    try:
        from rag_lab.retrieval.semantic_cache import cache_stats
        cs = cache_stats()
        print(f"[OK] Semantic cache      : {cs['total_entries']} entry(s) across {len(cs['workspaces'])} workspace(s)")
    except Exception as exc:
        print(f"[!!] Semantic cache error: {exc}")
        ok = False

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from rag_lab.security.access import PolicyIntegrityError, authenticate_admin, has_admin

    signer_parent = argparse.ArgumentParser(add_help=False)
    signer_parent.add_argument(
        "--as",
        dest="signer",
        default=argparse.SUPPRESS,
        metavar="ADMIN",
        help="Administrator performing the workspace change (prompts for password).",
    )

    parser = argparse.ArgumentParser(description="Manage multi-tenant workspaces and inspect system status.")
    parser.add_argument(
        "--as",
        dest="signer",
        default=None,
        metavar="ADMIN",
        help="Administrator performing the workspace change (prompts for password).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all workspaces and their indexing statistics")
    sub.add_parser("status", help="Show system health, model integrity, workspaces, and audit status")

    create_p = sub.add_parser("create", parents=[signer_parent], help="Create a new workspace")
    create_p.add_argument("workspace", help="Workspace name")

    delete_p = sub.add_parser(
        "delete",
        parents=[signer_parent],
        help="Permanently delete a workspace, its indexes, checkpoints, and grants",
    )
    delete_p.add_argument("workspace", help="Workspace name")
    delete_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args(argv)

    if args.command == "status":
        return print_status()

    if args.command == "list":
        workspaces = list_workspaces()
        if not workspaces:
            print("No workspaces found.")
            return 0
        for ws in workspaces:
            st = get_workspace_stats(ws)
            fts = "indexed" if st["bm25_indexed"] else "unindexed"
            print(f"{ws}: {st['files']} file(s), {st['chunks']} chunk(s) [{fts}]")
        return 0

    try:
        actor: str | None = None
        if args.signer:
            actor = authenticate_admin(args.signer)
        elif has_admin():
            raise PermissionError(
                "Creating or deleting workspaces needs an administrator. "
                "Re-run with --as <admin account> (make targets: ADMIN=<account>)."
            )

        if args.command == "create":
            ws_id = create_workspace(args.workspace, actor=actor)
            if not ws_id:
                print("Error: Invalid workspace name.", file=sys.stderr)
                return 1
            print(f"Created workspace '{ws_id}'.")
            return 0

        if args.command == "delete":
            ws_id = sanitize_workspace_id(args.workspace)
            if not ws_id:
                print("Error: Invalid workspace name.", file=sys.stderr)
                return 1
            if not args.yes:
                ans = input(f"Are you sure you want to permanently delete workspace '{ws_id}'? [y/N]: ").strip().lower()
                if ans not in ("y", "yes"):
                    print("Deletion cancelled.")
                    return 0
            delete_workspace(ws_id, actor=actor)
            print(f"Deleted workspace '{ws_id}'.")
            return 0
    except (ValueError, PolicyIntegrityError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
