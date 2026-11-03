"""Thread-safe, offline-cached Cross-Encoder Reranker using ONNX Runtime.

Loads pinned models (e.g. Xenova/ms-marco-MiniLM-L-6-v2) verified by SHA-256
hash in model_integrity.py without network access.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag_lab.config import settings
from rag_lab.security.model_integrity import verify_model

logger = logging.getLogger("reranker")

_RERANKER_LOCK = threading.Lock()


_RERANKERS: dict[tuple[str, str], TextCrossEncoder] = {}


def get_reranker(
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> TextCrossEncoder:
    """Return a verified TextCrossEncoder loaded completely offline, from exactly the files that were verified.

    Concurrent first calls load the model once: the check is repeated under the lock.
    """
    model_name = settings.reranker_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
    key = (model_name, cache_dir)
    cached = _RERANKERS.get(key)
    if cached is not None:
        return cached
    with _RERANKER_LOCK:
        cached = _RERANKERS.get(key)
        if cached is None:
            snapshot = verify_model(model_name, cache_dir)
            cached = TextCrossEncoder(
                model_name=model_name, cache_dir=cache_dir, specific_model_path=str(snapshot)
            )
            _RERANKERS[key] = cached
        return cached


def _reset_reranker_cache() -> None:
    with _RERANKER_LOCK:
        _RERANKERS.clear()


get_reranker.cache_clear = _reset_reranker_cache  # type: ignore[attr-defined]


def rerank_with_status(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_n: int = 3,
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Like rerank_documents, but also reports why the cross-encoder was skipped after a failure.

    The status is None when reranking ran, or was switched off on purpose. It is a short message
    when the cross-encoder failed and the initial ranking was used instead.
    """
    model_name = settings.reranker_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
    if not candidates:
        return [], None
    if not settings.enable_reranker or len(candidates) <= 1 or not query or not str(query).strip():
        return list(candidates[:top_n]), None

    try:
        reranker = get_reranker(model_name=model_name, cache_dir=cache_dir)
        docs = [c["content"] for c in candidates]
        scores = list(reranker.rerank(str(query).strip(), docs))

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n], None
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, falling back to initial ranking: %s", exc)
        return list(candidates[:top_n]), f"rerank_unavailable: {type(exc).__name__}; results use the initial ranking"


def rerank_documents(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_n: int = 3,
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Score and re-order candidate documents using cross-encoder attention over (query, document).

    If reranking is disabled or fails, cleanly returns candidates[:top_n] as fallback.
    """
    model_name = settings.reranker_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
    return rerank_with_status(query, candidates, top_n, model_name, cache_dir)[0]


def score_pairs(
    query: str,
    docs: Sequence[str],
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> list[float]:
    """Score (query, doc) text pairs using the pinned cross-encoder model.

    Returns raw logits for each document in docs.
    Returns an empty list if ENABLE_RERANKER is False, query is empty/whitespace,
    docs contains no valid text, or if scoring fails.
    """
    model_name = settings.reranker_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
    if not settings.enable_reranker or not query or not str(query).strip() or not docs:
        return []
    try:
        valid_docs = [str(d).strip() for d in docs if d is not None and str(d).strip()]
        if not valid_docs:
            return []
        reranker = get_reranker(model_name=model_name, cache_dir=cache_dir)
        return [float(s) for s in reranker.rerank(str(query).strip(), valid_docs)]
    except Exception as exc:
        logger.warning("Cross-encoder score_pairs failed: %s", exc)
        return []
