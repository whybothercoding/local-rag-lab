from contextlib import closing
import pickle
import sqlite3
from pathlib import Path
import pytest

from rag_lab.config import settings

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


# --- keyword-index tokenizer ------------------------------------------------------------------

def _index(tmp_path, chunks):
    from rag_lab.retrieval.sparse_search import index_sparse_chunks

    ids = [f"c{i}" for i in range(len(chunks))]
    index_sparse_chunks("ws", ids, [f"doc{i}.md" for i in range(len(chunks))], chunks, base_dir=str(tmp_path))


def _sources(tmp_path, query):
    from rag_lab.retrieval.sparse_search import search_sparse_chunks

    return [r["citation"].split("] ", 1)[1] for r in search_sparse_chunks("ws", query, top_n=3, base_dir=str(tmp_path))]


def test_index_uses_the_configured_tokenizer_and_rebuilds_when_it_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sparse_tokenizer", "porter")
    _index(tmp_path, ["Rollback SLA"])
    monkeypatch.setattr(settings, "sparse_tokenizer", "trigram")
    _index(tmp_path, ["Rollback SLA"])
    with closing(sqlite3.connect(tmp_path / "ws.db")) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'").fetchone()[0]
    assert "trigram" in schema and "porter" not in schema


def test_trigram_matches_compounds_inflections_and_accents(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sparse_tokenizer", "trigram")
    _index(tmp_path, [
        "Nach der Probezeit gilt eine Kündigungsfrist von 2 Monaten.",
        "Η τηλεργασία επιτρέπεται έως 2 ημέρες την εβδομάδα.",
        "Standard production deployments run on Tuesday.",
    ])
    assert _sources(tmp_path, "Frist")[0] == "doc0.md"            # part of a German compound
    assert _sources(tmp_path, "kundigungsfrist")[0] == "doc0.md"  # without the umlaut
    assert _sources(tmp_path, "τηλεργασίας")[0] == "doc1.md"      # another Greek inflection
    assert _sources(tmp_path, "deployment")[0] == "doc2.md"       # English singular
    assert _sources(tmp_path, "an")[:1] == []                     # too short for trigrams: no error, no match


def test_unknown_tokenizer_is_refused(tmp_path, monkeypatch):
    import pytest as _pytest

    from rag_lab.retrieval.sparse_search import SparseIndexError

    monkeypatch.setattr(settings, "sparse_tokenizer", "stemmer9000")
    with _pytest.raises(SparseIndexError, match="SPARSE_TOKENIZER"):
        _index(tmp_path, ["x"])


def test_changing_the_tokenizer_forces_a_reindex(tmp_path, monkeypatch):
    from rag_lab.ingestion.ingest import ingest_workspace

    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("Rollback SLA is 5 minutes.", encoding="utf-8")
    assert ingest_workspace("tok", str(source)).ok
    assert ingest_workspace("tok", str(source)).unchanged is True
    monkeypatch.setattr(settings, "sparse_tokenizer", "unicode61")
    result = ingest_workspace("tok", str(source))
    assert result.ok and result.unchanged is False
