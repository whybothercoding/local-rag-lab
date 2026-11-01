#!/usr/bin/env python3
"""Parse a folder of source documents into one workspace index.

Dense vectors go to ChromaDB and keyword vectors go to a BM25 pickle.
Embeddings are computed before either store is replaced, so a failed
embedding call leaves the previous index in place.
"""
import argparse
import getpass
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

import workspace_manager
from access import CLEARANCE_LEVELS, DEFAULT_CLEARANCE, is_admin, sign_in
from audit import record as audit_record
from config import MAX_SOURCE_BYTES
from embedding_provider import embedding_fingerprint, get_embedding_model
from parsers import SUPPORTED_EXTENSIONS, extract_text_from_file, unsupported_reason
from sparse_search import index_sparse_chunks
from workspace_manager import ensure_base_dirs, sanitize_workspace_id

logger = logging.getLogger("ingest")

# Chroma rejects a single add() above a few thousand rows.
_ADD_BATCH = 256


@dataclass
class IngestResult:
    workspace_id: str
    documents: int = 0
    chunks: int = 0
    indexed_files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unchanged: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.chunks > 0 or self.unchanged)


def _get_manifest_path(ws_id: str) -> Path:
    return Path(workspace_manager.BM25_DIR) / f"{ws_id}_manifest.json"


def _load_manifest(ws_id: str) -> dict | None:
    path = _get_manifest_path(ws_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_manifest(ws_id: str, manifest_data: dict) -> None:
    path = _get_manifest_path(ws_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save manifest for %s: %s", ws_id, exc)


def _add_in_batches(collection, ids, documents, metadatas, embeddings) -> None:
    for start in range(0, len(ids), _ADD_BATCH):
        end = start + _ADD_BATCH
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


def _delete_quiet(client, name: str) -> None:
    try:
        client.delete_collection(name=name)
    except Exception as exc:
        logger.info("Collection %s was not deleted (%s)", name, exc)


def _publish_collection(client, name: str, ids, documents, metadatas, embeddings) -> None:
    """Write a side collection first, then replace the live collection.

    The side collection is removed only after the live copy has been written.
    If the live write fails, the side collection still holds the new vectors.
    """
    build_name = f"{name}__build"
    _delete_quiet(client, build_name)
    building = client.create_collection(name=build_name)
    _add_in_batches(building, ids, documents, metadatas, embeddings)

    _delete_quiet(client, name)
    try:
        live = client.create_collection(name=name)
        _add_in_batches(live, ids, documents, metadatas, embeddings)
    except Exception:
        logger.exception("Live collection %s failed; vectors remain in %s", name, build_name)
        raise
    _delete_quiet(client, build_name)


def _extract_doc_clearance(raw_text: str) -> tuple[str, str]:
    """Extract clearance tag from YAML frontmatter or document header, returning (clearance, body)."""
    clearance = DEFAULT_CLEARANCE
    text = raw_text
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, re.DOTALL)
    if fm_match:
        try:
            parsed = yaml.safe_load(fm_match.group(1))
            if isinstance(parsed, dict) and "clearance" in parsed:
                candidate = str(parsed["clearance"]).strip().lower()
                if candidate in CLEARANCE_LEVELS:
                    clearance = candidate
                text = raw_text[fm_match.end():]
        except Exception:
            pass
    else:
        hdr_match = re.match(
            r"^(?:#\s*clearance:\s*|\[clearance\]:\s*)([a-zA-Z0-9_-]+)\s*\n",
            raw_text,
            re.IGNORECASE,
        )
        if hdr_match:
            candidate = hdr_match.group(1).strip().lower()
            if candidate in CLEARANCE_LEVELS:
                clearance = candidate
            text = raw_text[hdr_match.end():]
    return clearance, text


def _collect_documents(source_path: Path) -> tuple[list[dict], list[str]]:
    raw_docs: list[dict] = []
    skipped: list[str] = []
    files = sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.name.startswith(".")
    )
    for file_path in files:
        relative = str(file_path.relative_to(source_path))
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(f"{relative}: {unsupported_reason(ext)}")
            continue
        if file_path.stat().st_size > MAX_SOURCE_BYTES:
            skipped.append(f"{relative}: larger than {MAX_SOURCE_BYTES} bytes")
            continue
        try:
            raw_content = extract_text_from_file(file_path)
        except Exception as exc:
            skipped.append(f"{relative}: {exc}")
            continue
        if not raw_content:
            skipped.append(f"{relative}: no extractable text (a scanned PDF has no text layer)")
            continue
        clearance, content = _extract_doc_clearance(raw_content)
        raw_docs.append({
            "content": content,
            "clearance": clearance,
            "source": relative,
            "doc_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    return raw_docs, skipped


def ingest_workspace(
    workspace_id: str,
    source_dir: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    force: bool = False,
    actor: str = "system",
) -> IngestResult:
    """Index every supported file under source_dir into one workspace.

    Re-running replaces that workspace's Chroma collection and BM25 file.
    Other workspaces are left untouched.
    """
    ws_id = sanitize_workspace_id(workspace_id)
    result = IngestResult(workspace_id=ws_id)
    if not ws_id:
        result.error = "Invalid workspace name."
        return result

    ensure_base_dirs()
    source_path = Path(source_dir)
    if not source_path.is_dir():
        result.error = f"Directory not found: {source_dir}"
        return result

    raw_docs, skipped = _collect_documents(source_path)
    result.skipped = skipped
    if not raw_docs:
        result.error = f"No supported documents with extractable text found in {source_dir}"
        return result

    file_hashes = {doc["source"]: doc["doc_hash"] for doc in raw_docs}

    # Content-Addressable idempotency check: bypass re-indexing if files match
    manifest = _load_manifest(ws_id)
    if not force and manifest is not None:
        if (
            manifest.get("files") == file_hashes
            and manifest.get("chunk_size") == chunk_size
            and manifest.get("chunk_overlap") == chunk_overlap
            and manifest.get("embedding") == embedding_fingerprint()
        ):
            try:
                client = chromadb.PersistentClient(path=workspace_manager.CHROMA_PATH)
                existing_cols = [getattr(c, "name", str(c)) for c in client.list_collections()]
                fts_exists = os.path.exists(os.path.join(workspace_manager.BM25_DIR, f"{ws_id}.db"))
                if f"ws_{ws_id}" in existing_cols and fts_exists:
                    result.documents = len(raw_docs)
                    result.chunks = manifest.get("total_chunks", 0)
                    result.indexed_files = list(file_hashes.keys())
                    result.unchanged = True
                    logger.info("Workspace '%s' unchanged; skipping redundant embedding.", ws_id)
                    return result
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[str] = []
    metadatas: list[dict] = []
    chunk_ids: list[str] = []
    for doc_idx, doc in enumerate(raw_docs):
        doc_hash_short = doc["doc_hash"][:12]
        doc_clearance = doc.get("clearance", DEFAULT_CLEARANCE)
        for chunk_idx, chunk_text in enumerate(splitter.split_text(doc["content"])):
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
            cid = f"{ws_id}_{doc_idx}_{chunk_idx}_{chunk_hash}"
            chunks.append(chunk_text)
            metadatas.append({
                "source": doc["source"],
                "workspace": ws_id,
                "clearance": doc_clearance,
                "id": cid,
                "chunk_hash": chunk_hash,
                "doc_hash": doc_hash_short,
            })
            chunk_ids.append(cid)

    if not chunks:
        result.error = "Documents were read but produced no chunks."
        return result

    try:
        embeddings = get_embedding_model()
        vectors = embeddings.embed_documents(chunks)
    except Exception as exc:
        logger.exception("Embedding failed for workspace %s", ws_id)
        result.error = (
            f"Embedding calculation failed: {exc}. "
            "The existing index was not changed."
        )
        return result

    if len(vectors) != len(chunks):
        result.error = "Embedding service returned a different number of vectors than chunks."
        return result

    chroma_path = workspace_manager.CHROMA_PATH
    bm25_dir = workspace_manager.BM25_DIR
    collection_name = f"ws_{ws_id}"
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        _publish_collection(client, collection_name, chunk_ids, chunks, metadatas, vectors)
    except Exception as exc:
        logger.exception("Chroma write failed for workspace %s", ws_id)
        result.error = f"Vector index write failed: {exc}"
        return result

    # 2. Write SQLite FTS5 safe keyword index
    try:
        sources = [m.get("source", "document") for m in metadatas]
        clearances = [m.get("clearance", DEFAULT_CLEARANCE) for m in metadatas]
        index_sparse_chunks(ws_id, chunk_ids, sources, chunks, clearances=clearances, base_dir=bm25_dir)
    except Exception as exc:
        logger.exception("FTS5 keyword write failed for workspace %s", ws_id)
        result.error = f"Keyword index write failed: {exc}"
        return result

    # 3. Save ingestion manifest for idempotent verification
    _save_manifest(ws_id, {
        "workspace_id": ws_id,
        "files": file_hashes,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding": embedding_fingerprint(),
        "total_chunks": len(chunks),
        "total_docs": len(raw_docs),
    })

    result.documents = len(raw_docs)
    result.chunks = len(chunks)
    result.indexed_files = [doc["source"] for doc in raw_docs]
    audit_record(
        "index_workspace",
        workspace_id=ws_id,
        session_id="",
        data={"actor": actor, "documents": result.documents, "chunks": result.chunks},
    )
    logger.info("Indexed %s chunks from %s files into %s", result.chunks, result.documents, ws_id)
    return result


def authenticate_admin(account: str) -> str:
    """Sign an administrator in with their password. Indexing is an administrative action."""
    password = getpass.getpass(f"Password for admin account '{account}': ")
    result = sign_in(account, password)
    if not result.ok:
        raise SystemExit(f"{result.describe()} Indexing refused.")
    if not is_admin(result.username):
        raise SystemExit(f"'{result.username}' is not an administrator. Indexing refused.")
    return result.username


def _print_result(result: IngestResult) -> None:
    for note in result.skipped:
        print(f"Skipped {note}")
    if result.error:
        print(f"Error: {result.error}")
        return
    if result.unchanged:
        print(
            f"Workspace '{result.workspace_id}' is already up-to-date: "
            f"{result.documents} file(s), {result.chunks} chunks (skipped re-embedding)."
        )
        return
    print(
        f"Workspace '{result.workspace_id}' indexed successfully: "
        f"{result.documents} file(s), {result.chunks} chunks."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Multi-tenant knowledge ingestion")
    parser.add_argument("--workspace", required=False, help="Workspace name, for example accounting")
    parser.add_argument("--dir", required=False, help="Directory of source documents")
    parser.add_argument("--all", action="store_true", help="Auto-discover and sync all workspaces in knowledge_base/")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if files are unchanged")
    parser.add_argument("--as", dest="admin", required=True, metavar="ACCOUNT",
                        help="Administrator account performing the indexing (prompts for its password)")
    args = parser.parse_args()
    actor = authenticate_admin(args.admin)

    if args.all or not args.workspace:
        from workspace_manager import sync_all_workspaces
        print("Discovering and syncing all workspaces in knowledge_base/...")
        outcomes = sync_all_workspaces(force=args.force, actor=actor)
        if not outcomes:
            print("No workspace directories found under knowledge_base/.")
            sys.exit(0)
        for ws, outcome in outcomes.items():
            _print_result(outcome)
        sys.exit(0 if all(o.ok for o in outcomes.values()) else 1)

    if not args.dir:
        print("Error: --dir is required when specifying a single --workspace.")
        sys.exit(1)

    outcome = ingest_workspace(args.workspace, args.dir, force=args.force, actor=actor)
    _print_result(outcome)
    if not outcome.ok:
        sys.exit(1)
