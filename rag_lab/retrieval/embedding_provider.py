"""In-process and Ollama embedding adapters for dense vector retrieval.

Defaults to an in-process ONNX runtime (`fastembed`) requiring zero external daemons or network
calls. Setting EMBEDDING_PROVIDER=ollama switches to the local Ollama embedding model instead.

There is no automatic fallback between the two: they produce vectors of different sizes, so a
silent switch would index or query with a model the stored vectors were not built with. Changing
provider is a deliberate setting followed by a re-index (`make sync FORCE=1`). The fingerprint of
the model that built an index is stored with it, and retrieval refuses to query across a mismatch.
"""
from __future__ import annotations

import logging
from typing import Any
from langchain_core.embeddings import Embeddings

from rag_lab.security.model_integrity import ModelIntegrityError, verify_model

from rag_lab.config import settings

logger = logging.getLogger("embedding_provider")

_FASTEMBED_CACHE: dict[str, Any] = {}

PROVIDERS = ("fastembed", "ollama")
# Collection metadata key recording which embedding model built a workspace's vectors.
FINGERPRINT_KEY = "embedding_fingerprint"


class EmbeddingUnavailableError(RuntimeError):
    """The configured embedding model cannot be loaded. The message says how to fix it."""


def _active_provider(provider: str | None) -> str:
    active = (provider or settings.embedding_provider).lower().strip()
    if active not in PROVIDERS:
        raise EmbeddingUnavailableError(
            f"Unknown EMBEDDING_PROVIDER '{active}'. Use one of: {', '.join(PROVIDERS)}."
        )
    return active


class LocalFastEmbeddings(Embeddings):
    """In-process ONNX embedding engine using fastembed."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        model_name = settings.fastembed_model if model_name is None else model_name
        self.model_name = model_name
        self.cache_dir = cache_dir or settings.models_dir

    @property
    def model(self):
        cache_key = f"{self.model_name}:{self.cache_dir}"
        if cache_key not in _FASTEMBED_CACHE:
            # Loaded on first use, then kept: the hash check runs once per process, not per query.
            try:
                verified_path = verify_model(self.model_name, self.cache_dir)
                from fastembed import TextEmbedding
                _FASTEMBED_CACHE[cache_key] = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    specific_model_path=str(verified_path),
                )
            except ModelIntegrityError as exc:
                raise EmbeddingUnavailableError(
                    f"Embedding model {self.model_name} is not available: {exc}. "
                    "Run 'make models', or set EMBEDDING_PROVIDER=ollama and re-index with 'make sync FORCE=1'."
                ) from exc
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"Embedding model {self.model_name} failed to load ({type(exc).__name__}: {exc}). "
                    "Run 'make verify-models', or set EMBEDDING_PROVIDER=ollama and re-index with 'make sync FORCE=1'."
                ) from exc
        return _FASTEMBED_CACHE[cache_key]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Models such as e5 expect a "passage: " / "query: " prefix (settings, from the model profile).
        vectors = self.model.embed([settings.embedding_passage_prefix + text for text in texts])
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = list(self.model.embed([settings.embedding_query_prefix + text]))
        if not vectors:
            return []
        return [float(x) for x in vectors[0]]


def embedding_fingerprint(provider: str | None = None) -> str:
    """Identify the embedding model that produces vectors under the current configuration.

    Stored in ingestion manifests and collection metadata so a model change forces re-indexing and
    retrieval can refuse to compare vectors from two different models.
    """
    if _active_provider(provider) == "ollama":
        return f"ollama:{settings.embed_model}"
    return f"fastembed:{settings.fastembed_model}"


def stored_fingerprint(collection_metadata: Any, manifest: dict | None = None) -> str | None:
    """The fingerprint of the model that built an index, or None when it cannot be told.

    Collections built before the fingerprint was stored in their metadata fall back to the
    ingestion manifest, which has recorded it from the start.
    """
    if isinstance(collection_metadata, dict) and collection_metadata.get(FINGERPRINT_KEY):
        return str(collection_metadata[FINGERPRINT_KEY])
    if isinstance(manifest, dict) and manifest.get("embedding"):
        return str(manifest["embedding"])
    return None


def get_embedding_model(provider: str | None = None) -> Embeddings:
    """Factory returning the configured embedding model instance.

    `LocalFastEmbeddings` by default (in-process ONNX, no daemon), or `OllamaEmbeddings` when the
    provider is 'ollama'. A FastEmbed load failure raises EmbeddingUnavailableError on first use;
    it never switches model silently.
    """
    if _active_provider(provider) == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    return LocalFastEmbeddings()
