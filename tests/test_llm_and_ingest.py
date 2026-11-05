from pathlib import Path
import pytest

from rag_lab.pipeline.llm_provider import DeterministicContextLLM, get_chat_llm
from rag_lab.ingestion.ingest import ingest_workspace
from rag_lab.config import settings


def test_deterministic_context_llm_extraction():
    llm = DeterministicContextLLM()
    prompt = (
        "Context:\n"
        "FAQ Question: What is the old SLA? Answer: 15 minutes.\n---\n"
        "The standard company rollback SLA is under 5 minutes strictly. "
        "Server updates happen on Saturday.\n\n"
        "Question: What is the standard company rollback SLA Answer window?\n\n"
        "Answer concisely based strictly on the context:"
    )
    result = llm.invoke(prompt)
    assert "under 5 minutes" in result.content.lower()


def test_deterministic_context_llm_empty_context():
    llm = DeterministicContextLLM()
    prompt = "Context:\n\nQuestion: What is the SLA?\n\nAnswer concisely based strictly on the context:"
    result = llm.invoke(prompt)
    assert "could not find relevant documentation" in result.content.lower()


def test_get_chat_llm_deterministic_mode():
    llm = get_chat_llm(provider="deterministic")
    assert isinstance(llm, DeterministicContextLLM)


def test_idempotent_ingestion_manifest(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    doc = source_dir / "test.txt"
    doc.write_text("Company expense policy states all receipts must be submitted within 30 days.")

    # Redirect storage dirs to tmp_path
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))

    # First ingestion
    res1 = ingest_workspace("test_idempotent", str(source_dir))
    assert res1.ok is True
    assert res1.unchanged is False
    assert res1.chunks >= 1

    # Second ingestion: identical source directory
    res2 = ingest_workspace("test_idempotent", str(source_dir))
    assert res2.ok is True
    assert res2.unchanged is True

    # Third ingestion: forced re-index
    res3 = ingest_workspace("test_idempotent", str(source_dir), force=True)
    assert res3.ok is True
    assert res3.unchanged is False


def test_embedding_fingerprint_and_no_silent_model_swap(mocker):
    import rag_lab.retrieval.embedding_provider as embedding_provider
    from rag_lab.retrieval.embedding_provider import EmbeddingUnavailableError, embedding_fingerprint, get_embedding_model

    assert embedding_fingerprint("fastembed").startswith("fastembed:")

    # A FastEmbed failure surfaces as an actionable error; it never swaps in a model of another vector size.
    mocker.patch.object(embedding_provider, "LocalFastEmbeddings", side_effect=EmbeddingUnavailableError("run make models"))
    with pytest.raises(EmbeddingUnavailableError):
        get_embedding_model("fastembed")


def test_get_chat_llm_auto_fallback_and_openai_provider(mocker):
    import rag_lab.pipeline.llm_provider as llm_provider
    from rag_lab.pipeline.llm_provider import OpenAICompatibleChat

    mocker.patch.object(llm_provider, "is_ollama_alive", return_value=False)
    auto_llm = get_chat_llm(provider="auto")
    assert isinstance(auto_llm, DeterministicContextLLM)

    openai_llm = get_chat_llm(provider="openai")
    assert isinstance(openai_llm, OpenAICompatibleChat)

