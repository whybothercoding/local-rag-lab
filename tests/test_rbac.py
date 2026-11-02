"""Tests for Zero Trust Document-Level Role-Based Access Control (RBAC).

Verifies:
1. Clearance taxonomy (public, internal, confidential, secret) and hierarchical inheritance.
2. Document frontmatter / header parsing during ingestion.
3. Metadata filtering in ChromaDB and SQLite FTS5 search.
4. Total retrieval isolation: users with lower clearance cannot retrieve or view chunks
   from higher clearance documents even with exact keyword matches.
"""
from pathlib import Path
import pytest

from access import (
    CLEARANCE_LEVELS,
    DEFAULT_CLEARANCE,
    add_user,
    allowed_clearances,
    get_user_clearance,
    issue_token,
    load_policy,
    set_grant,
    set_user_clearance,
)
from ingest import _extract_doc_clearance, ingest_workspace
from sparse_search import get_fts_db_path, index_sparse_chunks, search_sparse_chunks
from app import hybrid_retrieval_node


def test_clearance_taxonomy_and_hierarchy():
    assert DEFAULT_CLEARANCE == "public"
    assert CLEARANCE_LEVELS == ("public", "internal", "confidential", "secret")

    assert allowed_clearances("public") == ["public"]
    assert allowed_clearances("internal") == ["public", "internal"]
    assert allowed_clearances("confidential") == ["public", "internal", "confidential"]
    assert allowed_clearances("secret") == ["public", "internal", "confidential", "secret"]

    # Unknown or invalid fallback to public
    assert allowed_clearances("invalid_role") == ["public"]
    assert allowed_clearances("") == ["public"]


def test_access_user_clearance_management(tmp_path: Path):
    policy_path = str(tmp_path / "access_policy.json")
    key_path = str(tmp_path / "access.key")

    # Add user with default clearance
    add_user("alice", "Password123!@#", clearance="public", path=policy_path)
    assert get_user_clearance("alice", path=policy_path) == "public"

    # Add user with confidential clearance
    add_user("bob", "Password123!@#", clearance="confidential", path=policy_path)
    assert get_user_clearance("bob", path=policy_path) == "confidential"

    # Update clearance
    set_user_clearance("alice", "secret", path=policy_path)
    assert get_user_clearance("alice", path=policy_path) == "secret"

    with pytest.raises(ValueError, match="Invalid clearance"):
        set_user_clearance("alice", "top_secret_not_valid", path=policy_path)


def test_extract_doc_clearance():
    # YAML frontmatter
    doc1 = "---\nclearance: confidential\ntitle: Internal memo\n---\nConfidential financial report content."
    clr1, body1 = _extract_doc_clearance(doc1)
    assert clr1 == "confidential"
    assert body1 == "Confidential financial report content."

    # Markdown header
    doc2 = "# clearance: secret\nSecret server deployment keys."
    clr2, body2 = _extract_doc_clearance(doc2)
    assert clr2 == "secret"
    assert body2 == "Secret server deployment keys."

    # Bracket header
    doc3 = "[clearance]: internal\nInternal guidelines."
    clr3, body3 = _extract_doc_clearance(doc3)
    assert clr3 == "internal"
    assert body3 == "Internal guidelines."

    # Default / unlabelled document
    doc4 = "Regular public documentation text."
    clr4, body4 = _extract_doc_clearance(doc4)
    assert clr4 == "public"
    assert body4 == "Regular public documentation text."

    # Invalid explicit clearance fails closed instead of silently downgrading to public
    with pytest.raises(ValueError, match="invalid clearance"):
        _extract_doc_clearance("---\nclearance: top_secret\n---\nClassified content.")
    with pytest.raises(ValueError, match="invalid clearance"):
        _extract_doc_clearance("# clearance: restricted\nClassified content.")


def test_sparse_search_clearance_filtering(tmp_path: Path):
    ws_id = "test_clearance_ws"
    bm25_dir = str(tmp_path / "bm25")

    chunk_ids = ["c1", "c2", "c3"]
    sources = ["handbook.md", "finances.md", "keys.md"]
    chunks = [
        "Project roadmap and public guidelines for everyone",
        "Confidential quarterly profits and executive compensation",
        "Top secret database master encryption passphrase",
    ]
    clearances = ["public", "confidential", "secret"]

    index_sparse_chunks(
        workspace_id=ws_id,
        chunk_ids=chunk_ids,
        sources=sources,
        chunks=chunks,
        clearances=clearances,
        base_dir=bm25_dir,
    )

    # 1. Public user query
    pub_res = search_sparse_chunks(
        workspace_id=ws_id,
        query="database passphrase",
        allowed_clearances=["public"],
        base_dir=bm25_dir,
    )
    # The secret chunk matches the query words, but user only has public clearance!
    assert len(pub_res) == 0

    # 2. Confidential user query
    conf_res = search_sparse_chunks(
        workspace_id=ws_id,
        query="compensation profits",
        allowed_clearances=["public", "internal", "confidential"],
        base_dir=bm25_dir,
    )
    assert len(conf_res) == 1
    assert conf_res[0]["id"] == "c2"

    # Confidential user still cannot see secret
    conf_secret = search_sparse_chunks(
        workspace_id=ws_id,
        query="database passphrase",
        allowed_clearances=["public", "internal", "confidential"],
        base_dir=bm25_dir,
    )
    assert len(conf_secret) == 0

    # 3. Secret user query
    sec_res = search_sparse_chunks(
        workspace_id=ws_id,
        query="database passphrase",
        allowed_clearances=["public", "internal", "confidential", "secret"],
        base_dir=bm25_dir,
    )
    assert len(sec_res) == 1
    assert sec_res[0]["id"] == "c3"

    # 4. Empty allowed_clearances fails closed and returns no results
    empty_res = search_sparse_chunks(
        workspace_id=ws_id,
        query="database passphrase",
        allowed_clearances=[],
        base_dir=bm25_dir,
    )
    assert empty_res == []


def test_hybrid_retrieval_clearance_enforcement(tmp_path: Path, monkeypatch):
    """Integration test verifying end-to-end clearance filtering in hybrid_retrieval_node."""
    ws_id = "rbac_test"
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Public document
    (source_dir / "public.md").write_text("General company info and public holiday calendar.")
    # Secret document
    (source_dir / "secret.md").write_text(
        "---\nclearance: secret\n---\nProject Quantum master encryption secret key is XYZ-999."
    )

    chroma_dir = str(tmp_path / "chroma")
    bm25_dir = str(tmp_path / "bm25")
    policy_path = str(tmp_path / "access_policy.json")

    monkeypatch.setattr("workspace_manager.CHROMA_PATH", chroma_dir)
    monkeypatch.setattr("workspace_manager.BM25_DIR", bm25_dir)
    monkeypatch.setattr("app.CHROMA_PATH", chroma_dir)
    monkeypatch.setattr("app.BM25_DIR", bm25_dir)
    monkeypatch.setattr("config.ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("access.ACCESS_POLICY_PATH", policy_path)

    # Ingest workspace
    res = ingest_workspace(ws_id, str(source_dir), force=True)
    assert res.ok

    # Setup two users: Bob (public) and Alice (secret)
    add_user("bob", "Password123!@#", clearance="public", path=policy_path)
    set_grant("bob", ws_id, ["query"], path=policy_path)
    bob_token = issue_token("bob")

    add_user("alice", "Password123!@#", clearance="secret", path=policy_path)
    set_grant("alice", ws_id, ["query"], path=policy_path)
    alice_token = issue_token("alice")

    # Bob asks for the secret key
    bob_state = {
        "workspace_id": ws_id,
        "principal": "bob",
        "principal_token": bob_token,
        "sanitized_question": "What is the Project Quantum master encryption secret key?",
    }
    bob_out = hybrid_retrieval_node(bob_state)
    # Bob must NOT receive the secret document chunks
    for doc in bob_out.get("documents", []):
        assert "XYZ-999" not in doc

    # Alice asks for the secret key
    alice_state = {
        "workspace_id": ws_id,
        "principal": "alice",
        "principal_token": alice_token,
        "sanitized_question": "What is the Project Quantum master encryption secret key?",
    }
    alice_out = hybrid_retrieval_node(alice_state)
    # Alice has secret clearance and should receive the relevant chunk
    found_secret = any("XYZ-999" in doc for doc in alice_out.get("documents", []))
    assert found_secret is True


def test_semantic_cache_partitioned_by_clearance(tmp_path: Path, monkeypatch):
    """Ensure a secret-clearance cached answer is never served to a public-clearance user."""
    from app import SEMANTIC_CACHE, cache_key, cache_lookup_node

    policy_path = str(tmp_path / "access_policy.json")
    monkeypatch.setattr("config.ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("access.ACCESS_POLICY_PATH", policy_path)

    add_user("alice", "Password123!@#", clearance="secret", path=policy_path)
    add_user("bob", "Password123!@#", clearance="public", path=policy_path)

    SEMANTIC_CACHE.clear()
    query = "what is the master key"
    # Alice (secret) caches an answer
    SEMANTIC_CACHE[cache_key("rbac_test", query, "secret")] = "The master key is XYZ-999."

    # Alice hits the cache
    alice_lookup = cache_lookup_node({
        "workspace_id": "rbac_test",
        "principal": "alice",
        "sanitized_question": query,
    })
    assert alice_lookup["cache_hit"] is True
    assert "XYZ-999" in alice_lookup["generation"]

    # Bob (public) MUST miss the secret cache entry
    bob_lookup = cache_lookup_node({
        "workspace_id": "rbac_test",
        "principal": "bob",
        "sanitized_question": query,
    })
    assert bob_lookup["cache_hit"] is False


def test_hybrid_retrieval_chroma_where_fallback_still_filters_clearance(tmp_path: Path, mocker, monkeypatch):
    """When Chroma's where filter raises an exception, the Python post-filter must still enforce clearance."""
    policy_path = str(tmp_path / "access_policy.json")
    monkeypatch.setattr("config.ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("access.ACCESS_POLICY_PATH", policy_path)
    add_user("bob", "Password123!@#", clearance="public", path=policy_path)

    fake_collection = mocker.MagicMock()

    def query_side_effect(**kwargs):
        if "where" in kwargs:
            raise RuntimeError("legacy collection without where index")
        return {
            "documents": [[
                "Public holiday schedule and general office hours.",
                "Secret master key XYZ-999 for production vault.",
            ]],
            "metadatas": [[
                {"source": "public.md", "clearance": "public"},
                {"source": "secret.md", "clearance": "secret"},
            ]],
            "ids": [["c_pub", "c_sec"]],
            "distances": [[0.1, 0.05]],
        }

    fake_collection.query.side_effect = query_side_effect
    fake_client = mocker.MagicMock()
    fake_client.get_collection.return_value = fake_collection
    mocker.patch("app.chromadb.PersistentClient", return_value=fake_client)
    mocker.patch("app.search_sparse_chunks", return_value=[])

    out = hybrid_retrieval_node({
        "workspace_id": "legacy_ws",
        "principal": "bob",
        "sanitized_question": "What is the secret master key or office hours?",
    })
    for doc in out["documents"]:
        assert "XYZ-999" not in doc
    assert any("office hours" in doc for doc in out["documents"])


class _DummyBM25:
    def get_scores(self, tokenized_query):
        return [1.0, 2.0]


def test_legacy_pickle_sparse_search_filters_by_clearance(tmp_path: Path):
    import pickle

    bm25_dir = tmp_path / "bm25_legacy"
    bm25_dir.mkdir()
    pkl_path = bm25_dir / "legacy_ws.pkl"
    with open(pkl_path, "wb") as handle:
        pickle.dump(
            {
                "bm25": _DummyBM25(),
                "chunks": ["Public handbook guidelines", "Secret encryption key XYZ-999"],
                "metadatas": [
                    {"source": "handbook.md", "clearance": "public"},
                    {"source": "vault.md", "clearance": "secret"},
                ],
                "ids": ["id_pub", "id_sec"],
            },
            handle,
        )

    pub_only = search_sparse_chunks(
        "legacy_ws",
        "encryption key handbook",
        allowed_clearances=["public"],
        base_dir=str(bm25_dir),
    )
    assert [r["id"] for r in pub_only] == ["id_pub"]

    with_secret = search_sparse_chunks(
        "legacy_ws",
        "encryption key handbook",
        allowed_clearances=["public", "secret"],
        base_dir=str(bm25_dir),
    )
    assert {r["id"] for r in with_secret} == {"id_pub", "id_sec"}


