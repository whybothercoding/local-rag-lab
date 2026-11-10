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

import numpy as np
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_lab.storage.workspace_manager as workspace_manager
from rag_lab.security.access import CLEARANCE_LEVELS, DEFAULT_CLEARANCE, is_admin, prompt_second_factor, sign_in
from rag_lab.security.audit import record as audit_record
from rag_lab.config import settings
from rag_lab.security.context_sanitizer import scan_chunk
from rag_lab.retrieval.embedding_provider import (
    EmbeddingUnavailableError,
    embedding_fingerprint,
    get_embedding_model,
    stored_fingerprint,
)
from rag_lab.ingestion.parsers import PDF_PAGE_BREAK, SUPPORTED_EXTENSIONS, extract_text_from_file, unsupported_reason
from rag_lab.retrieval.citation import source_label
from rag_lab.retrieval.sparse_search import delete_sparse_index, has_sparse_index, index_sparse_chunks
from rag_lab.storage.vector_store import get_vector_store
from rag_lab.storage.workspace_manager import ensure_base_dirs, sanitize_workspace_id

logger = logging.getLogger("ingest")

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
    # Vectors taken over from the previous index (chunks whose text did not change) and chunks embedded afresh.
    reused: int = 0
    embedded: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and (self.chunks > 0 or self.unchanged or self.cleared)


# Bump when chunk text changes for the same input, so unchanged files are still re-indexed.
# Bump when parser or chunker output changes: every workspace then re-indexes its unchanged files on the next sync.
# v2: DOCX in document order with headings, table rows labelled by column. v3: PDFs chunked per page.
CHUNKER_VERSION = "headings-v3"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_document(text: str, splitter) -> list[str]:
    """Split by markdown headings, then by size; every chunk starts with the headings it sits under.

    A chunk cut from the middle of a long section otherwise loses what it is about ("Leave requests
    longer than two weeks..." without "# Leave Policy"), which hurts both retrieval and the answer.
    """
    sections: list[tuple[str, list[str]]] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in body):
            sections.append(("\n".join(f"{'#' * level} {title}" for level, title in path), list(body)))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [(lvl, title) for lvl, title in path if lvl < level] + [(level, match.group(2))]
        else:
            body.append(line)
    flush()

    chunks: list[str] = []
    for heading, lines in sections:
        for piece in splitter.split_text("\n".join(lines).strip()):
            chunks.append(f"{heading}\n{piece}" if heading else piece)
    return chunks


def _reusable_vectors(ws_id: str, fingerprint: str, manifest: dict | None, force: bool) -> dict[str, list[float]]:
    """Vectors of the live index by exact chunk text, for chunks that need not be embedded again.

    A vector depends only on the model and on the text it embedded (prefix included), so an unchanged chunk keeps
    its vector however the rest of the workspace changed. Reuse is limited to the same workspace and to an index
    built by the same model fingerprint with the same passage prefix; an older manifest that did not record the
    prefix is not trusted, and neither is an index that cannot be read in full. FORCE=1 recomputes everything.
    """
    if force or manifest is None:
        return {}
    if manifest.get("embedding") != fingerprint or manifest.get("passage_prefix") != settings.embedding_passage_prefix:
        return {}
    try:
        store = get_vector_store()
        if stored_fingerprint(store.collection_metadata(ws_id), manifest) != fingerprint:
            return {}
        found: dict[str, list[float]] = {}
        for text, vector in store.read_vectors(ws_id):
            array = np.asarray(vector, dtype=np.float64)
            if text and array.ndim == 1 and array.size and bool(np.isfinite(array).all()):
                found.setdefault(text, array.tolist())
        return found
    except Exception:
        logger.debug("Could not read the previous vectors of %s; embedding everything", ws_id, exc_info=True)
        return {}


def _get_manifest_path(ws_id: str) -> Path:
    return Path(settings.bm25_dir) / f"{ws_id}_manifest.json"


def _load_manifest(ws_id: str) -> dict | None:
    path = _get_manifest_path(ws_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded: dict = json.load(f)
                return loaded
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


def _clear_index(ws_id: str) -> None:
    """Remove every index artifact of a workspace: vectors, keywords, manifest, graph, and cached answers."""
    get_vector_store().delete(ws_id)
    delete_sparse_index(ws_id, base_dir=settings.bm25_dir)
    manifest = _get_manifest_path(ws_id)
    if manifest.exists():
        manifest.unlink()
    try:
        from rag_lab.retrieval.graph_store import delete_workspace_graph
        delete_workspace_graph(ws_id)
    except Exception:
        # Never fail silently: a purge that leaves a graph behind must at least be visible in the log.
        logger.warning("Knowledge graph for workspace %s could not be deleted; it may be stale",
                       ws_id, exc_info=True)
    from rag_lab.retrieval.semantic_cache import clear_workspace_cache

    clear_workspace_cache(ws_id)


def _is_indexed(ws_id: str) -> bool:
    return _get_manifest_path(ws_id).exists() or has_sparse_index(ws_id, base_dir=settings.bm25_dir)


def _folder_default(file_path: Path, root: Path, cache: dict[Path, str | None]) -> str | None:
    """Clearance from the nearest `.clearance` file between a document and the source root."""
    folder = file_path.parent
    while True:
        if folder not in cache:
            marker = folder / ".clearance"
            # The cache entry is written only after the marker validates: poisoning it with None on the
            # raise would let sibling files silently fall back to the parent's clearance (fail-open).
            level = None
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
        if file_path.stat().st_size > settings.max_source_bytes:
            skipped.append(f"{relative}: larger than {settings.max_source_bytes} bytes")
            continue
        try:
            raw_content = extract_text_from_file(file_path)
            if not raw_content.replace(PDF_PAGE_BREAK, "").strip():
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


def _chunks_with_pages(source: str, content: str, splitter) -> tuple[list[tuple[str, int | None]], list[int]]:
    """Chunks with the page each came from (PDFs only), plus the pages that had no text.

    A PDF is chunked page by page so a citation names an exact page; a sentence that runs across a page
    break is therefore split there. Other formats have no pages.
    """
    if not source.lower().endswith(".pdf"):
        return [(chunk, None) for chunk in chunk_document(content, splitter)], []
    pieces: list[tuple[str, int | None]] = []
    blank: list[int] = []
    for number, page in enumerate(content.split(PDF_PAGE_BREAK), 1):
        if not page.strip():
            blank.append(number)
            continue
        pieces.extend((chunk, number) for chunk in chunk_document(page, splitter))
    return pieces, blank


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

    # The model that will embed this sync. A misconfigured provider is reported, never raised to the caller.
    try:
        fingerprint = embedding_fingerprint()
    except EmbeddingUnavailableError as exc:
        result.error = str(exc)
        return result

    raw_docs, skipped = _collect_documents(source_path)

    # Also merge documents from the complementary workspace directory (knowledge_base <-> uploads)
    # when source_dir is one of the managed workspace roots, so UI uploads and CLI sync never overwrite
    # each other. When both hold the same relative path, the knowledge_base copy always wins, whichever
    # entry point started the sync, and the shadowed copy is reported.
    resolved_src = source_path.resolve()
    kb_root = Path(settings.knowledge_base_dir).resolve()
    up_root = Path(settings.uploads_dir).resolve()
    complementary_dirs: list[Path] = []
    source_is_uploads = resolved_src.parent == up_root
    if source_is_uploads:
        for disc_id, kb_dir in workspace_manager.discover_knowledge_base_workspaces():
            if disc_id == ws_id:
                complementary_dirs.append(Path(kb_dir))
    elif resolved_src.parent == kb_root:
        up_ws = Path(settings.uploads_dir) / ws_id
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
            and manifest.get("embedding") == fingerprint
            and manifest.get("passage_prefix") == settings.embedding_passage_prefix
            and manifest.get("sparse_tokenizer", "porter") == settings.sparse_tokenizer
            and manifest.get("chunker") == CHUNKER_VERSION
        ):
            try:
                fts_exists = os.path.exists(os.path.join(settings.bm25_dir, f"{ws_id}.db"))
                if get_vector_store().exists(ws_id) and fts_exists:
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
        pieces, blank_pages = _chunks_with_pages(doc["source"], doc["content"], splitter)
        if blank_pages:
            result.warnings.append(
                f"{doc['source']}: page(s) {', '.join(map(str, blank_pages))} have no text layer (scanned?) "
                "and were not indexed; OCR them first if they matter"
            )
        for chunk_idx, (chunk_text, page) in enumerate(pieces):
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
                **({"page": page} if page else {}),
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

    # Only chunks whose text is new are embedded; the rest keep the vector they already have (one call, fastembed
    # batches it internally: measured, other batch sizes, thread counts and worker processes were not faster).
    stored = _reusable_vectors(ws_id, fingerprint, manifest, force)
    to_embed = list(dict.fromkeys(text for text in chunks if text not in stored))
    try:
        fresh = get_embedding_model().embed_documents(to_embed) if to_embed else []
    except Exception as exc:
        logger.exception("Embedding failed for workspace %s", ws_id)
        result.error = (
            f"Embedding calculation failed: {exc}. "
            "The existing index was not changed."
        )
        return result

    if len(fresh) != len(to_embed):
        result.error = "Embedding service returned a different number of vectors than chunks."
        return result
    by_text = {**stored, **dict(zip(to_embed, fresh))}
    vectors = [by_text[text] for text in chunks]
    result.reused = sum(1 for text in chunks if text in stored)
    result.embedded = len(to_embed)

    bm25_dir = settings.bm25_dir
    try:
        get_vector_store().publish(ws_id, chunk_ids, chunks, metadatas, vectors, fingerprint)
    except Exception as exc:
        logger.exception("Chroma write failed for workspace %s", ws_id)
        result.error = f"Vector index write failed: {exc}"
        return result

    # 2. Write SQLite FTS5 safe keyword index
    try:
        sources = [source_label(m.get("source", "document"), m.get("page")) for m in metadatas]
        clearances = [m.get("clearance", DEFAULT_CLEARANCE) for m in metadatas]
        index_sparse_chunks(ws_id, chunk_ids, sources, chunks, clearances=clearances, base_dir=bm25_dir)
    except Exception as exc:
        logger.exception("FTS5 keyword write failed for workspace %s", ws_id)
        result.error = f"Keyword index write failed: {exc}"
        return result

    # 2b. Build and save Knowledge Graph for multi-hop relationship retrieval
    try:
        from rag_lab.ingestion.graph_builder import build_workspace_graph
        from rag_lab.retrieval.graph_store import get_graph_path

        kg = build_workspace_graph(
            ws_id, [{"id": cid, "text": c} for cid, c in zip(chunk_ids, chunks)]
        )
        kg.save(get_graph_path(ws_id))
    except Exception as exc:
        logger.warning("Knowledge graph build failed for workspace %s: %s", ws_id, exc)

    # 3. Save ingestion manifest for idempotent verification
    try:
        _save_manifest(ws_id, {
            "workspace_id": ws_id,
            "files": file_hashes,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding": fingerprint,
            "passage_prefix": settings.embedding_passage_prefix,
            "sparse_tokenizer": settings.sparse_tokenizer,
            "chunker": CHUNKER_VERSION,
            "total_chunks": len(chunks),
            "total_docs": len(raw_docs),
        })
    except OSError as exc:
        logger.exception("Manifest write failed for workspace %s", ws_id)
        result.error = f"Index manifest write failed: {exc}"
        return result

    # 4. Prune stale cache entries from previous index versions for this workspace
    try:
        from rag_lab.retrieval.semantic_cache import prune_stale_cache
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
    result = sign_in(account, password, code_prompt=prompt_second_factor)
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
    reuse = f" ({result.reused} stored vector(s) reused, {result.embedded} embedded)" if result.reused else ""
    print(
        f"Workspace '{result.workspace_id}' indexed successfully: "
        f"{result.documents} file(s), {result.chunks} chunks{reuse}."
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

    if args.dir and not args.workspace:
        parser.error("--dir needs --workspace: without one it would silently sync every workspace")
    if args.all or not args.workspace:
        from rag_lab.storage.workspace_manager import sync_all_workspaces
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
        kb_candidate = Path(settings.knowledge_base_dir) / ws_id
        up_candidate = Path(settings.uploads_dir) / ws_id
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
