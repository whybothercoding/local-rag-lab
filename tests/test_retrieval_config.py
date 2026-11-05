"""Configurable retrieval candidate widths (dense, sparse, fusion pool, rerank top k)."""
import os
from unittest.mock import MagicMock

from rag_lab.config import Settings, settings
from rag_lab.pipeline.retrieval import hybrid_retrieval_node
from rag_lab.security.access import issue_token


def test_retrieval_widths_default_values():
    assert settings.retrieval_dense_k == 5
    assert settings.retrieval_sparse_k == 5
    assert settings.retrieval_fusion_top_n == 8
    assert settings.retrieval_rerank_top_k == 3


def test_retrieval_widths_read_from_environment(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DENSE_K", "12")
    monkeypatch.setenv("RETRIEVAL_SPARSE_K", "10")
    monkeypatch.setenv("RETRIEVAL_FUSION_TOP_N", "16")
    monkeypatch.setenv("RETRIEVAL_RERANK_TOP_K", "6")

    custom = Settings.from_env()
    assert custom.retrieval_dense_k == 12
    assert custom.retrieval_sparse_k == 10
    assert custom.retrieval_fusion_top_n == 16
    assert custom.retrieval_rerank_top_k == 6


def test_hybrid_retrieval_passes_configured_widths(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_dense_k", 9)
    monkeypatch.setattr(settings, "retrieval_sparse_k", 7)
    monkeypatch.setattr(settings, "retrieval_fusion_top_n", 14)
    monkeypatch.setattr(settings, "retrieval_rerank_top_k", 4)

    mock_store = MagicMock()
    mock_store.collection_metadata.return_value = None
    mock_store.query.return_value = []
    monkeypatch.setattr("rag_lab.pipeline.retrieval.get_vector_store", lambda: mock_store)

    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    monkeypatch.setattr("rag_lab.pipeline.retrieval.get_embedding_model", lambda: mock_embed)

    mock_sparse = MagicMock(return_value=[])
    monkeypatch.setattr("rag_lab.pipeline.retrieval.search_sparse_chunks", mock_sparse)

    state = {
        "sanitized_question": "test query",
        "workspace_id": "test_ws",
        "principal_token": issue_token("admin_test"),
        "telemetry": {},
    }

    hybrid_retrieval_node(state)

    # Verify dense query called with k=9
    assert mock_store.query.call_args[0][2] == 9
    # Verify sparse search called with top_n=7
    assert mock_sparse.call_args[1]["top_n"] == 7
