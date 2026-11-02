"""Thread-safe, offline-cached Cross-Encoder Reranker using ONNX Runtime.

Loads pinned models (e.g. Xenova/ms-marco-MiniLM-L-6-v2) verified by SHA-256
hash in model_integrity.py without network access.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Sequence
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import ENABLE_RERANKER, MODELS_DIR, RERANKER_MODEL
from model_integrity import verify_model

logger = logging.getLogger("reranker")

_RERANKER_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_reranker(
    model_name: str = RERANKER_MODEL,
    cache_dir: str = MODELS_DIR,
) -> TextCrossEncoder:
    """Return a verified TextCrossEncoder instance loaded completely offline."""
    with _RERANKER_LOCK:
        verify_model(model_name, cache_dir)
        return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)


def rerank_documents(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_n: int = 3,
    model_name: str = RERANKER_MODEL,
    cache_dir: str = MODELS_DIR,
) -> list[dict[str, Any]]:
    """Score and re-order candidate documents using cross-encoder attention over (query, document).

    If reranking is disabled or fails, cleanly returns candidates[:top_n] as fallback.
    """
    if not candidates:
        return []
    if not ENABLE_RERANKER or len(candidates) <= 1 or not query or not str(query).strip():
        return list(candidates[:top_n])

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
        return scored_candidates[:top_n]
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, falling back to initial ranking: %s", exc)
        return list(candidates[:top_n])


def score_pairs(
    query: str,
    docs: Sequence[str],
    model_name: str = RERANKER_MODEL,
    cache_dir: str = MODELS_DIR,
) -> list[float]:
    """Score (query, doc) text pairs using the pinned cross-encoder model.

    Returns raw logits for each document in docs.
    Returns an empty list if ENABLE_RERANKER is False, query is empty/whitespace,
    docs contains no valid text, or if scoring fails.
    """
    if not ENABLE_RERANKER or not query or not str(query).strip() or not docs:
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
