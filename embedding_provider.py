"""In-process and remote embedding adapters for dense vector retrieval.

Defaults to an in-process ONNX runtime (`fastembed`) requiring zero external daemons
or network calls. Seamlessly falls back to `langchain-ollama` if configured or requested.
"""
from __future__ import annotations

import logging
from typing import Any
from langchain_core.embeddings import Embeddings

from config import (
    EMBED_MODEL,
    EMBEDDING_PROVIDER,
    MODELS_DIR,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger("embedding_provider")

_FASTEMBED_CACHE: dict[str, Any] = {}


class LocalFastEmbeddings(Embeddings):
    """In-process ONNX embedding engine using fastembed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or MODELS_DIR

    @property
    def model(self):
        cache_key = f"{self.model_name}:{self.cache_dir}"
        if cache_key not in _FASTEMBED_CACHE:
            from fastembed import TextEmbedding
            _FASTEMBED_CACHE[cache_key] = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
        return _FASTEMBED_CACHE[cache_key]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.embed(texts)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = list(self.model.embed([text]))
        if not vectors:
            return []
        return vectors[0].tolist()


def get_embedding_model(provider: str | None = None) -> Embeddings:
    """Factory returning the configured embedding model instance.

    Uses `LocalFastEmbeddings` by default (zero daemons, pure Python in-process),
    or `OllamaEmbeddings` if provider is set to 'ollama'.
    """
    active_provider = (provider or EMBEDDING_PROVIDER).lower().strip()

    if active_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    try:
        return LocalFastEmbeddings()
    except Exception as exc:
        logger.warning(
            "FastEmbed initialization failed (%s); falling back to OllamaEmbeddings.",
            exc,
        )
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
