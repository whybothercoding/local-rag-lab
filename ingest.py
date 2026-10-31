#!/usr/bin/env python3
"""Parse a folder of source documents into one workspace index.

Dense vectors go to ChromaDB and keyword vectors go to a BM25 pickle.
Embeddings are computed before either store is replaced, so a failed
embedding call leaves the previous index in place.
"""
import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

import workspace_manager
from config import MAX_SOURCE_BYTES
from embedding_provider import get_embedding_model
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
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.chunks > 0


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


def _collect_documents(source_path: Path) -> tuple[list[dict], list[str]]:
    raw_docs: list[dict] = []
    skipped: list[str] = []
    files = sorted(
        path for path in source_path.rglob("*") if path.is_file() and not path.name.startswith(".")
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
            content = extract_text_from_file(file_path)
        except Exception as exc:
            skipped.append(f"{relative}: {exc}")
            continue
        if not content:
            skipped.append(f"{relative}: no extractable text (a scanned PDF has no text layer)")
            continue
        raw_docs.append({"content": content, "source": relative})
    return raw_docs, skipped


def ingest_workspace(
    workspace_id: str,
    source_dir: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[str] = []
    metadatas: list[dict] = []
    chunk_ids: list[str] = []
    for doc_idx, doc in enumerate(raw_docs):
        for chunk_idx, chunk_text in enumerate(splitter.split_text(doc["content"])):
            cid = f"{ws_id}_{doc_idx}_{chunk_idx}"
            chunks.append(chunk_text)
            metadatas.append({"source": doc["source"], "workspace": ws_id, "id": cid})
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
        index_sparse_chunks(ws_id, chunk_ids, sources, chunks, base_dir=bm25_dir)
    except Exception as exc:
        logger.exception("FTS5 keyword write failed for workspace %s", ws_id)
        result.error = f"Keyword index write failed: {exc}"
        return result

    result.documents = len(raw_docs)
    result.chunks = len(chunks)
    result.indexed_files = [doc["source"] for doc in raw_docs]
    logger.info("Indexed %s chunks from %s files into %s", result.chunks, result.documents, ws_id)
    return result


def _print_result(result: IngestResult) -> None:
    for note in result.skipped:
        print(f"Skipped {note}")
    if result.error:
        print(f"Error: {result.error}")
        return
    print(
        f"Workspace '{result.workspace_id}' indexed successfully: "
        f"{result.documents} file(s), {result.chunks} chunks."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Multi-tenant knowledge ingestion")
    parser.add_argument("--workspace", required=True, help="Workspace name, for example accounting")
    parser.add_argument("--dir", required=True, help="Directory of source documents")
    args = parser.parse_args()

    outcome = ingest_workspace(args.workspace, args.dir)
    _print_result(outcome)
    if not outcome.ok:
        sys.exit(1)
