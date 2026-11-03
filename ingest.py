#!/usr/bin/env python3
"""Parse a folder of source documents into one workspace index.

Dense vectors go to ChromaDB and keywords go to a SQLite FTS5 index. Embeddings are computed
before either store is replaced, so a failed embedding call leaves the previous index in place.

Clearance comes from, in order: a `clearance:` tag in a file's front matter or first line, and a
`.clearance` file in the file's folder (or any parent up to the source root). When both exist the
higher level wins. Untagged files are public. A malformed or unrecognized tag rejects the file.
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
from context_sanitizer import scan_chunk
from embedding_provider import embedding_fingerprint, get_embedding_model
from parsers import SUPPORTED_EXTENSIONS, extract_text_from_file, unsupported_reason
from sparse_search import delete_sparse_index, has_sparse_index, index_sparse_chunks
from workspace_manager import ensure_base_dirs, sanitize_workspace_id

logger = logging.getLogger("ingest")

# Chroma rejects a single add() above a few thousand rows.
_ADD_BATCH = 256
# Cosine distance keeps similarity meaningful whichever embedding model produced the vectors.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_HEADER_TAG = re.compile(r"\A(?:#\s*clearance:\s*|\[clearance\]:\s*)([a-zA-Z0-9_-]+)[ \t]*(?:\r?\n|\Z)", re.IGNORECASE)


@dataclass
class IngestResult:
    workspace_id: str
    documents: int = 0
    chunks: int = 0
    indexed_files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unchanged: bool = False
    cleared: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.chunks > 0 or self.unchanged or self.cleared)


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
    """Write the manifest atomically. A failure is raised: without a manifest the cache cannot tell index versions apart."""
    path = _get_manifest_path(ws_id)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


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
    """Build the new collection beside the live one, then swap them by renaming.

    Readers see the old index until the swap, and the gap while it happens is two renames long.
    If the swap fails the old collection is put back, so a failed publish never loses the live index.
    """
    build_name, old_name = f"{name}__build", f"{name}__old"
    _delete_quiet(client, build_name)
    _delete_quiet(client, old_name)
    building = client.create_collection(name=build_name, metadata=_COLLECTION_METADATA)
    _add_in_batches(building, ids, documents, metadatas, embeddings)

    live_exists = name in [getattr(c, "name", str(c)) for c in client.list_collections()]
    if live_exists:
        client.get_collection(name).modify(name=old_name)
    try:
        building.modify(name=name)
    except Exception:
        logger.exception("Could not publish collection %s; restoring the previous index", name)
        if live_exists:
            client.get_collection(old_name).modify(name=name)
        raise
    _delete_quiet(client, old_name)


def _clear_index(ws_id: str) -> None:
    """Remove every index artifact of a workspace: vectors, keywords, manifest, and cached answers."""
    client = chromadb.PersistentClient(path=workspace_manager.CHROMA_PATH)
    for name in (f"ws_{ws_id}", f"ws_{ws_id}__build", f"ws_{ws_id}__old"):
        _delete_quiet(client, name)
    delete_sparse_index(ws_id, base_dir=workspace_manager.BM25_DIR)
    manifest = _get_manifest_path(ws_id)
    if manifest.exists():
        manifest.unlink()
    from semantic_cache import clear_workspace_cache

    clear_workspace_cache(ws_id)


def _is_indexed(ws_id: str) -> bool:
    return _get_manifest_path(ws_id).exists() or has_sparse_index(ws_id, base_dir=workspace_manager.BM25_DIR)


def _folder_default(file_path: Path, root: Path, cache: dict[Path, str | None]) -> str | None:
    """Clearance from the nearest `.clearance` file between a document and the source root."""
    folder = file_path.parent
    while True:
        if folder not in cache:
            marker = folder / ".clearance"
            cache[folder] = None
            if marker.is_file() and not marker.is_symlink():
                lines = [ln.strip() for ln in marker.read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.strip().startswith("#")]
                level = lines[0].lower() if lines else ""
                if level not in CLEARANCE_LEVELS:
                    raise ValueError(
                        f"{marker} names clearance '{level}' (allowed: {', '.join(CLEARANCE_LEVELS)})"
                    )
                cache[folder] = level
        if cache[folder] is not None:
            return cache[folder]
        if folder == root or folder.parent == folder:
            return None
        folder = folder.parent


def _higher(first: str, second: str | None) -> str:
    if second is None:
        return first
    return max(first, second, key=CLEARANCE_LEVELS.index)


def _check_level(candidate: str) -> str:
    level = candidate.strip().lower()
    if level not in CLEARANCE_LEVELS:
        raise ValueError(f"invalid clearance '{level}' (allowed: {', '.join(CLEARANCE_LEVELS)})")
    return level


def _extract_doc_clearance(raw_text: str) -> tuple[str, str]:
    """Extract clearance tag from YAML front matter or document header, returning (clearance, body).

    Fail-closed: an unrecognized level raises ValueError, and so does front matter that is not valid
    YAML but visibly declares a clearance (it must not silently become public). Keys are matched
    case-insensitively.
    """
    front = _FRONT_MATTER.match(raw_text)
    if front:
        block = front.group(1)
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            if re.search(r"(?im)^\s*clearance\s*:", block):
                raise ValueError("front matter declares a clearance but is not valid YAML; fix it") from None
            parsed = None
        if isinstance(parsed, dict):
            lowered = {str(key).strip().lower(): value for key, value in parsed.items()}
            if "clearance" in lowered:
                return _check_level(str(lowered["clearance"])), raw_text[front.end():]
        return DEFAULT_CLEARANCE, raw_text
    header = _HEADER_TAG.match(raw_text)
    if header:
        return _check_level(header.group(1)), raw_text[header.end():]
    return DEFAULT_CLEARANCE, raw_text


def _collect_documents(source_path: Path) -> tuple[list[dict], list[str]]:
    raw_docs: list[dict] = []
    skipped: list[str] = []
    files = sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.name.startswith(".")
    )
    folder_cache: dict[Path, str | None] = {}
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
            if not raw_content:
                skipped.append(f"{relative}: no extractable text (a scanned PDF has no text layer)")
                continue
            clearance, content = _extract_doc_clearance(raw_content)
            clearance = _higher(clearance, _folder_default(file_path, source_path, folder_cache))
        except Exception as exc:
            skipped.append(f"{relative}: {exc}")
            continue
        raw_docs.append({
            "content": content,
            "clearance": clearance,
            "source": relative,
            # Hash the text as written plus the effective clearance, so changing only a tag (or a folder's
            # .clearance file) re-indexes the file.
            "doc_hash": hashlib.sha256(f"{clearance}\0{raw_content}".encode("utf-8")).hexdigest(),
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

    # Also merge documents from the complementary workspace directory (knowledge_base <-> uploads)
    # when source_dir is one of the managed workspace roots, so UI uploads and CLI sync never overwrite
    # each other. When both hold the same relative path, the knowledge_base copy always wins, whichever
    # entry point started the sync, and the shadowed copy is reported.
    resolved_src = source_path.resolve()
    kb_root = Path(workspace_manager.KNOWLEDGE_BASE_DIR).resolve()
    up_root = Path(workspace_manager.UPLOADS_DIR).resolve()
    complementary_dirs: list[Path] = []
    source_is_uploads = resolved_src.parent == up_root
    if source_is_uploads:
        for disc_id, kb_dir in workspace_manager.discover_knowledge_base_workspaces():
            if disc_id == ws_id:
                complementary_dirs.append(Path(kb_dir))
    elif resolved_src.parent == kb_root:
        up_ws = Path(workspace_manager.UPLOADS_DIR) / ws_id
        if up_ws.is_dir():
            complementary_dirs.append(up_ws)

    groups: list[tuple[str, list[dict]]] = [("uploads" if source_is_uploads else "knowledge_base", raw_docs)]
    for comp_dir in complementary_dirs:
        if comp_dir.is_dir() and comp_dir.resolve() != resolved_src:
            comp_docs, comp_skipped = _collect_documents(comp_dir)
            skipped.extend(comp_skipped)
            groups.append(("knowledge_base" if source_is_uploads else "uploads", comp_docs))
    groups.sort(key=lambda group: 0 if group[0] == "knowledge_base" else 1)

    merged: dict[str, dict] = {}
    for origin, docs in groups:
        for doc in docs:
            if doc["source"] in merged:
                skipped.append(f"{doc['source']}: shadowed by the knowledge_base copy; the {origin} copy was not indexed")
            else:
                merged[doc["source"]] = doc
    raw_docs = sorted(merged.values(), key=lambda d: d["source"])
    result.skipped = skipped
    if not raw_docs:
        if _is_indexed(ws_id):
            if not force:
                result.error = (
                    f"No supported documents found in {source_dir}, but '{ws_id}' is still indexed. "
                    "If every document was removed on purpose, re-run with --force (make sync FORCE=1) to purge the index."
                )
                return result
            try:
                _clear_index(ws_id)
            except Exception as exc:
                logger.exception("Could not purge index for %s", ws_id)
                result.error = f"Could not purge the index: {exc}"
                return result
            audit_record("index_cleared", workspace_id=ws_id, session_id="", data={"actor": actor})
            result.cleared = True
            return result
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
                logger.debug("Could not confirm the existing index for %s; re-indexing", ws_id, exc_info=True)

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
            safe, reason = scan_chunk(chunk_text)
            if not safe:
                result.warnings.append(
                    f"{doc['source']}: chunk {chunk_idx} contains an injected-instruction pattern ({reason}); "
                    "it will be withheld from answers"
                )
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
    try:
        _save_manifest(ws_id, {
            "workspace_id": ws_id,
            "files": file_hashes,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding": embedding_fingerprint(),
            "total_chunks": len(chunks),
            "total_docs": len(raw_docs),
        })
    except OSError as exc:
        logger.exception("Manifest write failed for workspace %s", ws_id)
        result.error = f"Index manifest write failed: {exc}"
        return result

    # 4. Prune stale cache entries from previous index versions for this workspace
    try:
        from semantic_cache import prune_stale_cache
        prune_stale_cache(ws_id)
    except Exception as exc:
        logger.debug("Could not prune stale cache for %s: %s", ws_id, exc)

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
    for note in result.warnings:
        print(f"Warning: {note}")
    if result.error:
        print(f"Error: {result.error}")
        return
    if result.cleared:
        print(f"Workspace '{result.workspace_id}' had no documents left; its index was purged.")
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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Multi-tenant knowledge ingestion")
    parser.add_argument("--workspace", required=False, help="Workspace name, for example accounting")
    parser.add_argument("--dir", required=False, help="Directory of source documents")
    parser.add_argument("--all", action="store_true", help="Auto-discover and sync all workspaces in knowledge_base/")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if files are unchanged")
    parser.add_argument("--as", dest="admin", required=True, metavar="ACCOUNT",
                        help="Administrator account performing the indexing (prompts for its password)")
    args = parser.parse_args(argv)
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
        ws_id = sanitize_workspace_id(args.workspace)
        kb_candidate = Path(workspace_manager.KNOWLEDGE_BASE_DIR) / ws_id
        up_candidate = Path(workspace_manager.UPLOADS_DIR) / ws_id
        for disc_id, disc_path in workspace_manager.discover_knowledge_base_workspaces():
            if disc_id == ws_id:
                kb_candidate = Path(disc_path)
                break
        if kb_candidate.is_dir():
            args.dir = str(kb_candidate)
        elif up_candidate.is_dir():
            args.dir = str(up_candidate)
        else:
            print(f"Error: No directory found for workspace '{args.workspace}'. Specify DIR=<path> or place files in knowledge_base/{ws_id}/.")
            sys.exit(1)

    outcome = ingest_workspace(args.workspace, args.dir, force=args.force, actor=actor)
    _print_result(outcome)
    if not outcome.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
