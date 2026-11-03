from contextlib import closing
import pickle
import sqlite3
from pathlib import Path
import pytest

from rag_lab.retrieval.embedding_provider import LocalFastEmbeddings, get_embedding_model
from rag_lab.retrieval.sparse_search import (
    SparseIndexError,
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


def test_sparse_search_refuses_legacy_pickle_and_cleans_up(tmp_path: Path):
    ws_id = "legacy_ws"
    pkl_path = tmp_path / f"{ws_id}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"chunks": ["Audit log retention is seven years."]}, f)

    # .pkl is never treated as an active sparse index and is never unpickled
    assert not has_sparse_index(ws_id, base_dir=str(tmp_path))
    assert search_sparse_chunks(ws_id, "audit log retention", top_n=1, base_dir=str(tmp_path)) == []

    # delete_sparse_index still purges any leftover .pkl file
    delete_sparse_index(ws_id, base_dir=str(tmp_path))
    assert not pkl_path.exists()


def test_sparse_search_missing_clearance_column_raises_sparse_index_error(tmp_path: Path):
    ws_id = "old_schema_ws"
    db_path = tmp_path / f"{ws_id}.db"
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, source UNINDEXED, content);")
        conn.execute("INSERT INTO chunks_fts VALUES ('c1', 'doc.md', 'secret content');")
        conn.commit()

    with pytest.raises(SparseIndexError, match="no clearance column"):
        search_sparse_chunks(ws_id, "secret content", base_dir=str(tmp_path))
