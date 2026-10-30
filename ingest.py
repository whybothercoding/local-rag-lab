#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from workspace_manager import sanitize_workspace_id, ensure_base_dirs, CHROMA_PATH, BM25_DIR

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def extract_text_from_file(file_path: Path) -> str:
    """Extracts raw text content across supported file formats: .md, .txt, .pdf, .docx."""
    ext = file_path.suffix.lower()

    if ext in [".md", ".txt"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    elif ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        pages_text = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
        return "\n\n".join(pages_text)

    elif ext == ".docx":
        import docx
        doc = docx.Document(str(file_path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)

    return ""


def ingest_workspace(workspace_id: str, source_dir: str, chunk_size: int = 600, chunk_overlap: int = 100):
    """Parses .md, .txt, .pdf, and .docx files in source_dir, chunks them, and stores
    them in a dedicated Chroma collection and serialized BM25 index.
    """
    ws_id = sanitize_workspace_id(workspace_id)
    if not ws_id:
        print("Error: Invalid workspace name provided.")
        return

    ensure_base_dirs()
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Directory not found: {source_dir}")
        return

    # 1. Discover and load documents across all supported extensions
    supported_exts = {".md", ".txt", ".pdf", ".docx"}
    raw_docs = []

    for file_path in source_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            try:
                content = extract_text_from_file(file_path)
                if content:
                    raw_docs.append({"content": content, "source": file_path.name})
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    if not raw_docs:
        print(f"No valid documents (.md, .txt, .pdf, .docx) found in {source_dir}")
        return

    print(f"Indexing {len(raw_docs)} document(s) into workspace '{ws_id}'...")

    # 2. Chunk documents dynamically
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadatas = []
    chunk_ids = []

    for doc_idx, doc in enumerate(raw_docs):
        sub_chunks = splitter.split_text(doc["content"])
        for chunk_idx, chunk_text in enumerate(sub_chunks):
            cid = f"{ws_id}_{doc_idx}_{chunk_idx}"
            chunks.append(chunk_text)
            metadatas.append({"source": doc["source"], "workspace": ws_id, "id": cid})
            chunk_ids.append(cid)

    # 3. Dense Storage (ChromaDB)
    embeddings = OllamaEmbeddings(model="qwen2.5-coder:7b", base_url=OLLAMA_URL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection_name = f"ws_{ws_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=collection_name)

    vectors = embeddings.embed_documents(chunks)
    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas, embeddings=vectors)

    # 4. Sparse Storage (BM25)
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunks": chunks,
        "metadatas": metadatas,
        "ids": chunk_ids,
    }
    bm25_path = os.path.join(BM25_DIR, f"{ws_id}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"Workspace '{ws_id}' indexed successfully: {len(chunks)} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tenant Knowledge Ingestion CLI")
    parser.add_argument("--workspace", required=True, help="Unique identifier for the knowledge base")
    parser.add_argument("--dir", required=True, help="Directory containing documents (.md, .txt, .pdf, .docx)")
    args = parser.parse_args()

    ingest_workspace(args.workspace, args.dir)
