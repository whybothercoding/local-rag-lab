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

CHROMA_PATH = "./chroma_data"
BM25_DIR = "./bm25_indices"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ingest_workspace(workspace_id: str, source_dir: str, chunk_size: int = 600, chunk_overlap: int = 100):
    """Parses all documents in source_dir, chunks them, and stores
    them in a dedicated Chroma collection and serialized BM25 index.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Directory not found: {source_dir}")
        return

    # 1. Discover and load text/markdown files
    raw_docs = []
    for file_path in source_path.glob("**/*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_docs.append({"content": f.read(), "source": file_path.name})
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    if not raw_docs:
        print(f"No documents (.md, .txt) found in {source_dir}")
        return

    print(f"Loaded {len(raw_docs)} documents from {source_dir}. Chunking...")

    # 2. Chunk documents dynamically
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadatas = []
    chunk_ids = []

    for doc_idx, doc in enumerate(raw_docs):
        sub_chunks = splitter.split_text(doc["content"])
        for chunk_idx, chunk_text in enumerate(sub_chunks):
            cid = f"{workspace_id}_{doc_idx}_{chunk_idx}"
            chunks.append(chunk_text)
            metadatas.append({"source": doc["source"], "workspace": workspace_id, "id": cid})
            chunk_ids.append(cid)

    print(f"Generated {len(chunks)} chunks.")

    # 3. Dense Storage (ChromaDB)
    embeddings = OllamaEmbeddings(model="qwen2.5-coder:7b", base_url=OLLAMA_URL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Reset/recreate collection for this workspace
    collection_name = f"ws_{workspace_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=collection_name)

    print("Computing embeddings and indexing into ChromaDB...")
    vectors = embeddings.embed_documents(chunks)
    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas, embeddings=vectors)

    # 4. Sparse Storage (BM25)
    os.makedirs(BM25_DIR, exist_ok=True)
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunks": chunks,
        "metadatas": metadatas,
        "ids": chunk_ids,
    }
    bm25_path = os.path.join(BM25_DIR, f"{workspace_id}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"Successfully ingested workspace '{workspace_id}'.")
    print(f"- Chroma collection: {collection_name}")
    print(f"- BM25 index: {bm25_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tenant Knowledge Ingestion CLI")
    parser.add_argument("--workspace", required=True, help="Unique identifier for the knowledge base (e.g. eshop, law, notes)")
    parser.add_argument("--dir", required=True, help="Directory containing markdown or text files")
    args = parser.parse_args()

    ingest_workspace(args.workspace, args.dir)
