"""Tests for Cross-Encoder Reranker integration.

Verifies:
1. Pinned SHA-256 weight integrity of Xenova/ms-marco-MiniLM-L-6-v2.
2. Cross-encoder scoring and re-ranking accuracy on relevant vs distraction chunks.
3. Graceful degradation when candidates <= 1 or when reranker is disabled.
4. Offline verification in fetch_models.py.
"""

from rag_lab.config import settings
from rag_lab.security.model_integrity import PINNED_MODELS, verify_model
from rag_lab.retrieval.reranker import rerank_documents, score_pairs


def test_reranker_pinned_model_integrity():
    """Verify that the reranker model weights match pinned SHA-256 hashes offline."""
    path = verify_model(settings.reranker_model, settings.models_dir)
    assert path.is_dir()
    # The weights file is whichever the profile pins (model_int8.onnx by default): a fixed name only passed on machines
    # that also held another profile's reranker.
    weights = [name for name in PINNED_MODELS[settings.reranker_model]["files"] if name.endswith(".onnx")]
    assert weights and all((path / name).is_file() for name in weights)


def test_reranker_reorders_by_relevance():
    """Cross-encoder should score high-relevance chunks higher than off-topic distractions."""
    query = "How long do I have to return an item for a refund?"
    candidates = [
        {
            "id": "c1",
            "content": "We offer shipping via FedEx and UPS to all 50 states.",
            "citation": "[c1] shipping.md",
        },
        {
            "id": "c2",
            "content": "Items can be returned within 30 days of purchase for a full refund.",
            "citation": "[c2] returns.md",
        },
        {
            "id": "c3",
            "content": "Our headquarters is located in Austin, Texas.",
            "citation": "[c3] company.md",
        },
    ]

    reranked = rerank_documents(query, candidates, top_n=2)
    assert len(reranked) == 2
    # c2 directly answers the return window question and should be top ranked
    assert reranked[0]["id"] == "c2"
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_reranker_empty_and_single_candidate():
    assert rerank_documents("query", []) == []

    single = [{"id": "c1", "content": "one chunk", "citation": "[c1] doc.md"}]
    res = rerank_documents("query", single, top_n=3)
    assert len(res) == 1
    assert res[0]["id"] == "c1"


def test_reranker_graceful_fallback(monkeypatch):
    """If reranker throws an unexpected exception, it should fallback to top_n candidates."""
    candidates = [
        {"id": "c1", "content": "chunk 1", "citation": "[c1]"},
        {"id": "c2", "content": "chunk 2", "citation": "[c2]"},
    ]

    def faulty_get_reranker(*args, **kwargs):
        raise RuntimeError("ONNX memory error")

    monkeypatch.setattr("rag_lab.retrieval.reranker.get_reranker", faulty_get_reranker)
    res = rerank_documents("test query", candidates, top_n=1)
    assert len(res) == 1
    assert res[0]["id"] == "c1"


def test_score_pairs_returns_logits():
    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.",
        "Bananas grow abundantly on Mars.",
    ]
    scores = score_pairs(query, docs)
    assert len(scores) == 2
    assert isinstance(scores[0], float)
    assert isinstance(scores[1], float)
    assert scores[0] > scores[1]


def test_score_pairs_empty_inputs():
    assert score_pairs("", ["doc"]) == []
    assert score_pairs("   ", ["doc"]) == []
    assert score_pairs("query", []) == []
    assert score_pairs("query", ["", "   "]) == []
    # Blank and None documents keep their slot, scored as -inf: the result stays positionally
    # aligned with the input, so a caller can zip it without shifting scores onto other docs.
    res = score_pairs("query", ["", None, "valid doc"])
    assert len(res) == 3
    assert res[0] == res[1] == float("-inf")
    assert isinstance(res[2], float) and res[2] > float("-inf")


def test_reranker_whitespace_query_returns_candidates():
    candidates = [
        {"id": "c1", "content": "chunk 1", "citation": "[c1]"},
        {"id": "c2", "content": "chunk 2", "citation": "[c2]"},
    ]
    res = rerank_documents("   ", candidates, top_n=2)
    assert len(res) == 2
    assert res[0]["id"] == "c1"


def test_score_pairs_disabled_and_error(monkeypatch):
    monkeypatch.setattr(settings, "enable_reranker", False)
    assert score_pairs("query", ["doc"]) == []

    monkeypatch.setattr(settings, "enable_reranker", True)
    monkeypatch.setattr("rag_lab.retrieval.reranker.get_reranker", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("error")))
    assert score_pairs("query", ["doc"]) == []
