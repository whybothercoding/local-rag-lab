"""Embedding provider contract: no silent model switch, fingerprinted indexes, and a labelled mismatch."""
import chromadb
import httpx
import pytest
from langchain_ollama import OllamaEmbeddings

import embedding_provider
from app import BLOCKING_DEGRADED_PREFIXES, _embedding_mismatch, hybrid_retrieval_node
from config import EMBED_MODEL, FASTEMBED_MODEL
from embedding_provider import (
    FINGERPRINT_KEY,
    EmbeddingUnavailableError,
    LocalFastEmbeddings,
    embedding_fingerprint,
    get_embedding_model,
    stored_fingerprint,
)
from llm_provider import ollama_models

QUESTION = "What is the rollback SLA?"
OLLAMA_FINGERPRINT = f"ollama:{EMBED_MODEL}"
FASTEMBED_FINGERPRINT = f"fastembed:{FASTEMBED_MODEL}"


def _retrieval_state(question: str = QUESTION) -> dict:
    return {"sanitized_question": question, "workspace_id": "default", "telemetry": {}, "degraded": []}


# --- provider selection ---------------------------------------------------------

def test_fingerprints_name_the_provider_and_model():
    assert embedding_fingerprint("fastembed") == FASTEMBED_FINGERPRINT
    assert embedding_fingerprint("ollama") == OLLAMA_FINGERPRINT
    assert embedding_fingerprint(" Ollama ") == OLLAMA_FINGERPRINT


def test_ollama_provider_returns_ollama_embeddings():
    assert isinstance(get_embedding_model("ollama"), OllamaEmbeddings)
    assert isinstance(get_embedding_model("fastembed"), LocalFastEmbeddings)


def test_unknown_provider_is_refused_with_the_valid_choices():
    with pytest.raises(EmbeddingUnavailableError, match="fastembed, ollama"):
        get_embedding_model("openai")
    with pytest.raises(EmbeddingUnavailableError):
        embedding_fingerprint("sentence-transformers")


def test_fastembed_failure_raises_instead_of_switching_to_ollama(tmp_path):
    """A different model would produce vectors of another size: the failure must surface, not switch."""
    broken = LocalFastEmbeddings(cache_dir=str(tmp_path / "no-models-here"))
    with pytest.raises(EmbeddingUnavailableError, match="make models"):
        broken.embed_query("rollback")
    with pytest.raises(EmbeddingUnavailableError):
        broken.embed_documents(["rollback"])


def test_fastembed_load_errors_other_than_integrity_are_wrapped(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise OSError("disk unreadable")

    monkeypatch.setattr(embedding_provider, "verify_model", boom)
    with pytest.raises(EmbeddingUnavailableError, match="OSError: disk unreadable"):
        LocalFastEmbeddings(cache_dir=str(tmp_path)).embed_query("x")


def test_fastembed_model_is_loaded_once_per_process(monkeypatch):
    get_embedding_model("fastembed").embed_query("warm up")
    calls = []
    monkeypatch.setattr(embedding_provider, "verify_model", lambda *a, **k: calls.append(a))
    get_embedding_model("fastembed").embed_query("first")
    get_embedding_model("fastembed").embed_query("second")
    assert calls == [], "the hash check runs once per process; queries must not repeat it"


# --- stored fingerprints -------------------------------------------------------

def test_stored_fingerprint_prefers_collection_metadata_then_manifest():
    meta = {"hnsw:space": "cosine", FINGERPRINT_KEY: OLLAMA_FINGERPRINT}
    assert stored_fingerprint(meta, {"embedding": FASTEMBED_FINGERPRINT}) == OLLAMA_FINGERPRINT
    # Legacy collection without the key: the manifest has always recorded the model.
    assert stored_fingerprint({"hnsw:space": "cosine"}, {"embedding": FASTEMBED_FINGERPRINT}) == FASTEMBED_FINGERPRINT
    assert stored_fingerprint(None, None) is None
    assert stored_fingerprint({}, {}) is None


def test_ingest_stamps_the_fingerprint_on_the_live_collection(rollback_index):
    import app

    client = chromadb.PersistentClient(path=app.CHROMA_PATH)
    meta = client.get_collection("ws_default").metadata
    assert meta[FINGERPRINT_KEY] == embedding_fingerprint()
    assert meta["hnsw:space"] == "cosine"
    names = [c.name for c in client.list_collections()]
    assert "ws_default__build" not in names and "ws_default__old" not in names


# --- retrieval refuses to compare vectors from two models ----------------------

def test_provider_switch_without_reindex_skips_dense_with_a_labelled_note(rollback_index, monkeypatch):
    monkeypatch.setattr("app.embedding_fingerprint", lambda: OLLAMA_FINGERPRINT)

    def must_not_embed(*args, **kwargs):
        raise AssertionError("dense retrieval must not embed a query against a mismatched index")

    monkeypatch.setattr("app.get_embedding_model", must_not_embed)
    state = hybrid_retrieval_node(_retrieval_state())

    notes = [n for n in state["degraded"] if "re-index" in n]
    assert len(notes) == 1
    assert notes[0].startswith(BLOCKING_DEGRADED_PREFIXES), "a mismatch must block auto-approval"
    assert FASTEMBED_FINGERPRINT in notes[0] and OLLAMA_FINGERPRINT in notes[0]
    assert state["telemetry"]["dense_candidates_count"] == 0
    assert state["documents"], "keyword retrieval still answers from the index"


def test_matching_fingerprint_queries_dense_normally(rollback_index):
    state = hybrid_retrieval_node(_retrieval_state())
    assert not any(n.startswith("dense_retrieval_unavailable") for n in state["degraded"])
    assert state["telemetry"]["dense_candidates_count"] > 0


def test_legacy_collection_without_metadata_is_checked_against_the_manifest(rollback_index, monkeypatch):
    legacy_meta = {"hnsw:space": "cosine"}
    assert _embedding_mismatch("default", legacy_meta) is None
    monkeypatch.setattr("app.embedding_fingerprint", lambda: OLLAMA_FINGERPRINT)
    note = _embedding_mismatch("default", legacy_meta)
    assert note and note.startswith("dense_retrieval_unavailable")


def test_index_of_unknown_origin_is_queried(tmp_path, monkeypatch):
    """No metadata and no manifest: nothing to compare, so retrieval proceeds as before."""
    monkeypatch.setattr("app.BM25_DIR", str(tmp_path))
    monkeypatch.setattr("app.embedding_fingerprint", lambda: OLLAMA_FINGERPRINT)
    assert _embedding_mismatch("default", {"hnsw:space": "l2"}) is None
    (tmp_path / "default_manifest.json").write_text("not json", encoding="utf-8")
    assert _embedding_mismatch("default", None) is None


def test_unloadable_embedding_model_is_reported_with_its_fix(rollback_index, monkeypatch):
    class Broken:
        def embed_query(self, text):
            raise EmbeddingUnavailableError("Embedding model X is not available. Run 'make models'")

    monkeypatch.setattr("app.get_embedding_model", lambda: Broken())
    state = hybrid_retrieval_node(_retrieval_state())
    assert any(n.startswith("dense_retrieval_unavailable") and "make models" in n for n in state["degraded"])
    assert state["documents"]


def test_reindex_after_provider_switch_clears_the_mismatch(rollback_index, monkeypatch):
    """Re-syncing under the new provider stores the new fingerprint, so retrieval trusts the index again."""
    from ingest import ingest_workspace

    monkeypatch.setattr("ingest.embedding_fingerprint", lambda: OLLAMA_FINGERPRINT)
    assert ingest_workspace("default", str(rollback_index), force=True).ok
    monkeypatch.setattr("app.embedding_fingerprint", lambda: OLLAMA_FINGERPRINT)
    state = hybrid_retrieval_node(_retrieval_state())
    assert not any("re-index" in n for n in state["degraded"])


# --- Ollama model discovery (used by make status) ------------------------------

def test_ollama_models_lists_pulled_models_with_and_without_latest(monkeypatch):
    payload = {"models": [{"name": "nomic-embed-text:latest"}, {"model": "qwen2.5-coder:7b"}, {}]}
    monkeypatch.setattr("llm_provider.httpx.get", lambda *a, **k: httpx.Response(200, json=payload))
    names = ollama_models("http://ollama.invalid")
    assert {"nomic-embed-text", "nomic-embed-text:latest", "qwen2.5-coder:7b"} <= names


def test_ollama_models_is_none_when_unreachable_or_erroring(monkeypatch):
    def refused(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("llm_provider.httpx.get", refused)
    assert ollama_models("http://ollama.invalid") is None
    monkeypatch.setattr("llm_provider.httpx.get", lambda *a, **k: httpx.Response(500))
    assert ollama_models("http://ollama.invalid") is None
