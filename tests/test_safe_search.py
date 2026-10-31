import os
import pickle
from pathlib import Path
import pytest
from rank_bm25 import BM25Okapi

from embedding_provider import LocalFastEmbeddings, get_embedding_model
from sparse_search import (
    index_sparse_chunks,
    search_sparse_chunks,
    delete_sparse_index,
    has_sparse_index,
)


def test_in_process_fast_embeddings():
    embedder = LocalFastEmbeddings()
    vector = embedder.embed_query("Security and compliance policy")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert isinstance(vector[0], float)

    docs = ["First document text", "Second document text"]
    doc_vectors = embedder.embed_documents(docs)
    assert len(doc_vectors) == 2
    assert len(doc_vectors[0]) == len(vector)


def test_embedding_factory_default():
    embedder = get_embedding_model()
    assert isinstance(embedder, LocalFastEmbeddings)


def test_sqlite_fts5_index_and_search(tmp_path: Path):
    ws_id = "test_fts_ws"
    ids = ["c1", "c2", "c3"]
    sources = ["sla.md", "deploy.md", "hr.txt"]
    chunks = [
        "Company rollback SLA is strictly under 5 minutes.",
        "Production deployments occur on Tuesday evenings.",
        "Annual leave request requires manager signoff.",
    ]

    index_sparse_chunks(ws_id, ids, sources, chunks, base_dir=str(tmp_path))
    assert has_sparse_index(ws_id, base_dir=str(tmp_path))

    # Query for SLA
    results = search_sparse_chunks(ws_id, "What is the rollback SLA?", top_n=2, base_dir=str(tmp_path))
    assert len(results) >= 1
    assert results[0]["id"] == "c1"
    assert "strictly under 5 minutes" in results[0]["content"]
    assert "[c1] sla.md" in results[0]["citation"]

    # Delete index
    delete_sparse_index(ws_id, base_dir=str(tmp_path))
    assert not has_sparse_index(ws_id, base_dir=str(tmp_path))


def test_sparse_search_legacy_pickle_fallback(tmp_path: Path):
    ws_id = "legacy_ws"
    chunks = [
        "Audit log retention is seven years.",
        "Password expiration interval is 90 days.",
    ]
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    data = {
        "bm25": bm25,
        "chunks": chunks,
        "ids": ["leg_1", "leg_2"],
        "metadatas": [{"source": "audit.txt"}, {"source": "auth.txt"}],
    }

    pkl_path = tmp_path / f"{ws_id}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    assert has_sparse_index(ws_id, base_dir=str(tmp_path))

    results = search_sparse_chunks(ws_id, "audit log retention", top_n=1, base_dir=str(tmp_path))
    assert len(results) == 1
    assert results[0]["id"] == "leg_1"
    assert "seven years" in results[0]["content"]

    delete_sparse_index(ws_id, base_dir=str(tmp_path))
    assert not has_sparse_index(ws_id, base_dir=str(tmp_path))
