"""Hybrid retrieval with a relevance gate, context sanitizing, document grading, and query rewriting."""
from __future__ import annotations

import re

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import json
import time

from chromadb.errors import NotFoundError
import chromadb

from rag_lab.config import settings
from rag_lab.pipeline import llm_provider
from rag_lab.pipeline.llm_provider import DeterministicContextLLM
from rag_lab.retrieval.embedding_provider import EmbeddingUnavailableError
from rag_lab.retrieval.embedding_provider import embedding_fingerprint
from rag_lab.retrieval.embedding_provider import get_embedding_model
from rag_lab.retrieval.embedding_provider import stored_fingerprint
from rag_lab.retrieval.reranker import rerank_with_status, score_pairs
from rag_lab.retrieval.sparse_search import has_sparse_index
from rag_lab.retrieval.sparse_search import search_sparse_chunks
from rag_lab.security.access import allowed_clearances
from rag_lab.security.access import get_user_clearance
from rag_lab.security.access import verify_token
from rag_lab.security.audit import record as audit_record
from rag_lab.security.context_sanitizer import sanitize_chunks
from rag_lab.security.context_sanitizer import scan_chunk

from rag_lab.pipeline.state import StateInput
from rag_lab.pipeline.text import _content_terms, _coverage, _keywords, _stem_variants


def _similarity(distance: float, space: str) -> float:
    """Convert a Chroma distance to a cosine-like similarity for the collection's distance space.

    Collections built by this version use cosine distance. Older ones use squared L2 on unit vectors,
    where cosine = 1 - d / 2.
    """
    if space == "cosine":
        return 1.0 - distance
    return 1.0 - distance / 2.0


class _EmbeddingMismatch(RuntimeError):
    """Raised inside dense retrieval to skip it with a labelled note."""


def _manifest(ws_id: str) -> dict | None:
    """The workspace's ingestion manifest, or None. Used for indexes that predate fingerprinted collections."""
    path = Path(settings.bm25_dir) / f"{ws_id}_manifest.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _embedding_mismatch(ws_id: str, collection_metadata: Any) -> str | None:
    """A labelled note when the index was built by another embedding model than the runtime uses.

    Vectors from two models are not comparable (and usually differ in size), so dense retrieval is
    skipped instead of returning wrong neighbours or failing with an opaque dimension error.
    """
    built_with = stored_fingerprint(collection_metadata, _manifest(ws_id))
    runtime = embedding_fingerprint()
    if built_with is None or built_with == runtime:
        return None
    return (
        f"dense_retrieval_unavailable: the index was built with {built_with} but the runtime uses {runtime}; "
        "re-index with 'make sync FORCE=1' (or restore the model it was built with)"
    )


_CONJUNCTION = re.compile(r"\s*(?:,\s*)?\b(?:and|or|und|oder|et|ou|και|ή)\b\s*", re.IGNORECASE)


def _question_parts(question: str) -> list[str]:
    """The parts of a compound question, each with at least two content words; [] if it is not one."""
    parts = [p.strip(" ,;?") for p in _CONJUNCTION.split(question) if p.strip(" ,;?")]
    parts = [p for p in parts if len(_content_terms(p)) >= 2]
    return parts if len(parts) >= 2 else []


def _top_rerank_score(query: str, reranked: List[Dict[str, Any]], rerank_note: str | None) -> float | None:
    """The cross-encoder score of the best passage, or None when the reranker is off or failed."""
    if not reranked or rerank_note or not settings.enable_reranker:
        return None
    scores = [item["rerank_score"] for item in reranked if "rerank_score" in item]
    if scores:
        return float(max(scores))
    # A single candidate is not reranked; score it directly so the same gate applies.
    single = score_pairs(query, [reranked[0]["content"]])
    return single[0] if single else None


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """Merges and re-ranks dense and sparse candidate chunks using Reciprocal Rank Fusion.
    Score = sum(1.0 / (k + rank_i))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:top_n]]


def hybrid_retrieval_node(state: StateInput) -> dict[str, Any]:
    """Retrieve candidates, then keep them only if they are relevant enough to answer from.

    Relevance is judged two ways: the best cosine similarity from the vector store, and how many
    of the query's content words one chunk contains. If neither clears its threshold, no documents
    are passed on, and the graph rewrites the query and eventually abstains.
    """
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    telemetry = dict(state.get("telemetry", {}))
    degraded: List[str] = []

    dense_candidates: List[Dict[str, Any]] = []
    sparse_candidates: List[Dict[str, Any]] = []

    # 1. Resolve user clearance levels for Zero Trust document-level RBAC filtering
    principal = state.get("principal", "")
    if not principal:
        principal = verify_token(str(state.get("principal_token", "")))
    user_clr = get_user_clearance(principal) if principal else "public"
    authorized_clrs = allowed_clearances(user_clr)

    # 1. Dense retrieval (ChromaDB) with clearance filter + fail-closed post-query verification
    try:
        chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        collection = chroma_client.get_collection(name=f"ws_{ws_id}")
        raw_meta = getattr(collection, "metadata", None)
        space = str(raw_meta.get("hnsw:space", "l2")) if isinstance(raw_meta, dict) else "l2"
        mismatch = _embedding_mismatch(ws_id, raw_meta)
        if mismatch:
            raise _EmbeddingMismatch(mismatch)
        q_vec = get_embedding_model().embed_query(query)
        try:
            results = collection.query(
                query_embeddings=[q_vec],  # type: ignore[arg-type]  # chroma stubs want numpy; lists are accepted
                n_results=5,
                where={"clearance": {"$in": authorized_clrs}},  # type: ignore[dict-item]
            )
        except Exception:
            results = collection.query(query_embeddings=[q_vec], n_results=20)  # type: ignore[arg-type]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [2.0] * len(docs)
        for doc, meta, cid, distance in zip(docs, metas, ids, distances):
            meta_dict = meta or {}
            # A chunk with no clearance tag is treated as unreadable, never as public.
            doc_clr = str(meta_dict.get("clearance", "")).strip().lower()
            if not doc_clr or doc_clr not in authorized_clrs:
                continue
            dense_candidates.append({
                "id": cid,
                "content": doc,
                "citation": f"[{cid}] {meta_dict.get('source', 'document')}",
                "similarity": _similarity(float(distance), space),
            })
    except NotFoundError:
        degraded.append("not_indexed: this workspace has no vector index yet")
    except _EmbeddingMismatch as exc:
        degraded.append(str(exc))
    except EmbeddingUnavailableError as exc:
        degraded.append(f"dense_retrieval_unavailable: {exc}")
    except Exception as exc:
        degraded.append(f"dense_retrieval_unavailable: {type(exc).__name__}")

    # 2. Sparse retrieval (SQLite FTS5 with BM25 ranking, filtered by clearance, legacy fallback)
    try:
        sparse_candidates = search_sparse_chunks(
            ws_id,
            query,
            top_n=5,
            allowed_clearances=authorized_clrs,
            base_dir=settings.bm25_dir,
        )
        if dense_candidates and not has_sparse_index(ws_id, base_dir=settings.bm25_dir):
            degraded.append("sparse_retrieval_unavailable: the keyword index is missing; re-index the workspace")
    except Exception as exc:
        degraded.append(f"sparse_retrieval_unavailable: {type(exc).__name__}: {exc}")

    # 3. Retrieval signals, kept for telemetry and as the fallback relevance gate.
    candidates = dense_candidates + sparse_candidates
    best_similarity = max((c["similarity"] for c in dense_candidates), default=0.0)
    query_terms = _content_terms(query)
    best_coverage = max((_coverage(query_terms, c["content"]) for c in candidates), default=0.0)

    # 4. Fuse with Reciprocal Rank Fusion (RRF), drop chunks that carry injected instructions, and only
    # then cut to the final top 3 with the cross-encoder. Screening before the cut means a poisoned chunk
    # is replaced by the next-best clean one instead of leaving a gap.
    fused_docs = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=8) if candidates else []
    clean_pool: List[Dict[str, Any]] = []
    dropped_chunks: List[Dict[str, Any]] = []
    for item in fused_docs:
        is_safe, reason = scan_chunk(item["content"])
        if is_safe:
            clean_pool.append(item)
        else:
            dropped_chunks.append({
                "citation": item["citation"],
                "reason": reason or "unknown_violation",
                "preview": item["content"][:120],
            })
    reranked_docs, rerank_note = rerank_with_status(query, clean_pool, top_n=3) if clean_pool else ([], None)
    if rerank_note:
        degraded.append(rerank_note)

    # 5. Relevance gate. The cross-encoder reads the question and the passage together, so its score
    # separates "answers this" from "is merely about the same topic" far better than vector similarity,
    # across languages too. Without a reranker score, similarity or word coverage decide.
    best_rerank = _top_rerank_score(query, reranked_docs, rerank_note)
    if best_rerank is not None and settings.relevance_gate == "reranker":
        relevant = best_rerank >= settings.relevance_min_rerank_score
        telemetry["relevance_gate"] = "reranker"
        if not relevant:
            # A two-part question ("the dress code, and the rollback SLA?") scores low as a whole when the
            # documents cover only one part. Each part is scored on its own; the full question still goes to
            # the model, which says what the documents do not contain.
            for part in _question_parts(query):
                part_docs, part_note = rerank_with_status(part, clean_pool, top_n=3)
                part_score = _top_rerank_score(part, part_docs, part_note)
                if part_score is not None and part_score >= settings.relevance_min_rerank_score and part_score > best_rerank:
                    reranked_docs, best_rerank, relevant = part_docs, part_score, True
                    telemetry["answered_part"] = part
    else:
        relevant = bool(candidates) and (
            best_similarity >= settings.relevance_min_similarity or best_coverage >= settings.relevance_min_coverage
        )
        telemetry["relevance_gate"] = "similarity"
    if not relevant:
        # Nothing reaches generation, so nothing was released from these chunks either.
        reranked_docs, dropped_chunks = [], []
    elif best_rerank is not None and settings.relevance_gate == "reranker":
        # Cut weak passages: they dilute the context and give stated figures false support.
        floor = max(settings.relevance_min_rerank_score, best_rerank - settings.rerank_keep_margin)
        kept = [d for i, d in enumerate(reranked_docs) if i == 0 or d.get("rerank_score", best_rerank) >= floor]
        telemetry["passages_cut"] = len(reranked_docs) - len(kept)
        reranked_docs = kept

    chunks_found = [item["content"] for item in reranked_docs]
    citations = [item["citation"] for item in reranked_docs]

    telemetry["retrieval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["dense_candidates_count"] = len(dense_candidates)
    telemetry["sparse_candidates_count"] = len(sparse_candidates)
    telemetry["fused_chunks_count"] = len(chunks_found)
    telemetry["best_similarity"] = round(best_similarity, 4)
    telemetry["best_coverage"] = round(best_coverage, 4)
    telemetry["best_rerank_score"] = None if best_rerank is None else round(best_rerank, 4)
    telemetry["relevant"] = relevant

    audit_record(
        "retrieval",
        workspace_id=ws_id,
        session_id=state.get("session_id", ""),
        data={
            "citations": citations,
            "dense_candidates": len(dense_candidates),
            "sparse_candidates": len(sparse_candidates),
            "best_similarity": telemetry["best_similarity"],
            "best_coverage": telemetry["best_coverage"],
            "relevant": relevant,
            "degraded": degraded,
            "retry_count": state.get("retry_count", 0),
        },
    )

    return {
        "documents": chunks_found,
        "citations": citations,
        "telemetry": telemetry,
        "degraded": degraded,
        "dropped_chunks": dropped_chunks,
    }


def sanitize_context_node(state: StateInput) -> dict[str, Any]:
    """Inspect retrieved chunks for indirect prompt injection and drop poisoned content."""
    t0 = time.perf_counter()
    raw_docs = state.get("documents", [])
    raw_citations = state.get("citations", [])
    telemetry = dict(state.get("telemetry", {}))
    degraded = list(state.get("degraded", []))
    ws_id = state.get("workspace_id", "default")
    session_id = state.get("session_id", "")

    # Chunks already screened out during retrieval are reported here together with anything this
    # second pass still finds, so the audit trail and the warning cover both.
    clean_docs, clean_citations, late_dropped = sanitize_chunks(raw_docs, raw_citations)
    dropped = list(state.get("dropped_chunks") or []) + late_dropped

    telemetry["context_sanitizer_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["poisoned_chunks_dropped"] = len(dropped)

    if dropped:
        degraded.append(
            f"security_warning: dropped {len(dropped)} poisoned document chunk(s) containing indirect prompt injection payloads"
        )
        for violation in dropped:
            audit_record(
                "indirect_injection_detected",
                workspace_id=ws_id,
                session_id=session_id,
                data={
                    "citation": violation["citation"],
                    "reason": violation["reason"],
                    "preview": violation["preview"],
                },
            )

    return {
        "documents": clean_docs,
        "citations": clean_citations,
        "telemetry": telemetry,
        "degraded": degraded,
        "dropped_chunks": [],
    }


def grade_documents_node(state: StateInput) -> dict[str, Any]:
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if not docs:
        if retries >= settings.max_retrieval_retries:
            return {"grade_status": "circuit_break"}
        return {"grade_status": "rewrite", "retry_count": retries + 1}

    return {"grade_status": "approved"}


def _llm_keyword_query(question: str) -> str | None:
    """Ask a real model for a keyword search query. Returns None when only the extractive fallback is available."""
    try:
        llm = llm_provider.get_chat_llm()
        if llm_provider.llm_is_fallback(llm) or isinstance(llm, DeterministicContextLLM):
            return None
        prompt = (
            "Rewrite this question as one short keyword search query for a document index. "
            "Reply with the query only.\n\nQuestion: " + question
        )
        candidate = " ".join(str(llm.invoke(prompt).content).split())[:200]
        return candidate or None
    except Exception:
        return None


def query_rewrite_node(state: StateInput) -> dict[str, Any]:
    """Retry retrieval with a different query. Always derived from the user's original question.

    First retry: a model-written keyword query when a model is available, otherwise the question's
    content words without filler. Later retries add word-stem variants to widen the match.
    """
    base = state.get("original_question") or state.get("sanitized_question", "")
    attempt = int(state.get("retry_count", 0))
    words = _keywords(base)
    if attempt <= 1:
        rewritten = _llm_keyword_query(base) or " ".join(words)
    else:
        rewritten = " ".join(words + _stem_variants(words))
    # New documents are coming, so the previous answer's grader note no longer applies.
    return {"sanitized_question": rewritten or base, "grader_feedback": ""}
