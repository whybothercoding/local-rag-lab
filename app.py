import os
import re
import time
import hashlib
from typing import TypedDict, List, Dict, Any, Optional

import chromadb
from chromadb.errors import NotFoundError

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from pathlib import Path

import workspace_manager
from config import (
    DB_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBED_MODEL,
    CHROMA_PATH,
    BM25_DIR,
    RELEVANCE_MIN_COVERAGE,
    RELEVANCE_MIN_SIMILARITY,
)
from access import allowed_clearances, authorize, get_user_clearance, verify_token
from audit import record as audit_record
from embedding_provider import get_embedding_model
from llm_provider import DeterministicContextLLM, get_chat_llm
from redaction import redact_pii
from context_sanitizer import sanitize_chunks
from reranker import rerank_documents
from sparse_search import search_sparse_chunks


class GraphState(TypedDict, total=False):
    workspace_id: str
    session_id: str
    principal_token: str
    principal: str
    question: str
    original_question: str
    sanitized_question: str
    documents: List[str]
    citations: List[str]
    generation: str
    intent: str
    cache_hit: bool
    is_safe: bool
    grade_status: str
    hallucination_status: str
    relevance_status: str
    retry_count: int
    history: List[str]
    telemetry: Dict[str, Any]
    degraded: List[str]


StateInput = GraphState | dict[str, Any]
SEMANTIC_CACHE: dict[str, Any] = {}
SEMANTIC_CACHE_CITATIONS: dict[str, List[str]] = {}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ABSTAIN_MESSAGE = "I could not find relevant documentation in this workspace to answer your query."

STOPWORDS = frozenset({
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how", "does", "did", "have",
    "has", "had", "with", "from", "into", "about", "there", "their", "your", "this", "that", "these",
    "those", "must", "should", "would", "could", "under", "over", "between", "after", "before", "tell",
    "give", "show", "list", "they", "them", "were", "been", "being", "will", "shall", "each", "every",
    "some", "any", "details", "overview",
})
_WORD = re.compile(r"[a-z0-9]+")


def _content_terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) >= 4 and word not in STOPWORDS}


def _coverage(query_terms: set[str], chunk: str) -> float:
    """Share of the query's content words that appear in one chunk."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(_WORD.findall(chunk.lower()))) / len(query_terms)


def _index_version(ws_id: str) -> str:
    """Fingerprint of a workspace's indexed content. It changes whenever the workspace is re-indexed."""
    manifest = Path(workspace_manager.BM25_DIR) / f"{ws_id}_manifest.json"
    if not manifest.is_file():
        return "none"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def cache_key(ws_id: str, query: str, clearance: str = "public") -> str:
    """Cached answers are keyed by workspace, clearance, query, and index version, so re-indexing or clearance changes invalidate them."""
    return _sha256(f"{ws_id}:{clearance}:{_index_version(ws_id)}:{query.strip().lower()}")


# --- HELPER: RECIPROCAL RANK FUSION (RRF) ---

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


# --- NODES ---

def guardrail_input_node(state: StateInput) -> dict[str, Any]:
    # Callers must pass redacted input: the checkpoint stores the submitted state before this node runs.
    t0 = time.perf_counter()
    raw = state.get("question", "")
    ws_id = state.get("workspace_id", "default")
    session_id = state.get("session_id", "")
    # Identity comes only from a signed session token. A bare name in the input is ignored.
    principal = verify_token(str(state.get("principal_token", "")))
    telemetry = dict(state.get("telemetry", {}))

    # Authorization comes first: unauthorized callers get no processing and no retrieval.
    if not authorize(principal, ws_id, "query"):
        denied_query = redact_pii(raw)
        audit_record(
            "access_denied",
            workspace_id=ws_id,
            session_id=session_id,
            data={"principal": principal, "permission": "query", "query": denied_query},
        )
        return {
            "is_safe": False,
            "generation": "[ACCESS DENIED: this account is not authorized for the selected workspace.]",
            "sanitized_question": denied_query,
            "telemetry": telemetry,
        }

    adversarial_patterns = [
        r"ignore previous instructions",
        r"system prompt override",
        r"bypass security rules",
        r"disregard all prior directives",
    ]
    for pattern in adversarial_patterns:
        if re.search(pattern, raw, re.IGNORECASE):
            telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            safe_query = redact_pii(raw)
            audit_record(
                "query_received",
                workspace_id=ws_id,
                session_id=session_id,
                data={"status": "blocked", "reason": "prompt_injection", "query": safe_query},
            )
            return {
                "is_safe": False,
                "generation": "[SECURITY VIOLATION: Adversarial prompt injection detected.]",
                "sanitized_question": safe_query,
                "telemetry": telemetry,
            }

    sanitized = redact_pii(raw)
    audit_record(
        "query_received",
        workspace_id=ws_id,
        session_id=session_id,
        data={"status": "accepted", "query": sanitized},
    )

    telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "is_safe": True,
        "principal": principal,
        "original_question": sanitized,
        "sanitized_question": sanitized,
        "telemetry": telemetry,
    }


def cache_lookup_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
    clearance = get_user_clearance(principal) if principal else "public"
    telemetry = dict(state.get("telemetry", {}))
    key = cache_key(ws_id, query, clearance)

    telemetry["cache_lookup_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if key in SEMANTIC_CACHE:
        telemetry["cache_hit"] = True
        cached = SEMANTIC_CACHE[key]
        cached_citations = SEMANTIC_CACHE_CITATIONS.get(key, state.get("citations", []))
        audit_record(
            "cache_hit",
            workspace_id=ws_id,
            session_id=state.get("session_id", ""),
            data={"query": query, "answer_sha256": _sha256(str(cached))},
        )
        return {
            "cache_hit": True,
            "generation": str(cached),
            "citations": cached_citations,
            "telemetry": telemetry,
        }

    telemetry["cache_hit"] = False
    return {"cache_hit": False, "telemetry": telemetry}


def router_node(state: StateInput) -> dict[str, Any]:
    raw_query = state.get("sanitized_question", "").lower()
    cleaned = " ".join(_WORD.findall(raw_query))
    words = cleaned.split()
    chitchat_phrases = ("who are you", "what can you do", "thank you")
    chitchat_words = {"hello", "hi", "hey", "thanks"}

    if any(phrase in cleaned for phrase in chitchat_phrases):
        return {"intent": "chitchat"}
    domain_terms = _content_terms(cleaned) - chitchat_words
    if any(w in chitchat_words for w in words) and not domain_terms:
        return {"intent": "chitchat"}
    return {"intent": "rag"}


def chitchat_node(state: StateInput) -> dict[str, Any]:
    audit_record(
        "chitchat",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={"query": state.get("sanitized_question", "")},
    )
    return {
        "generation": "Hello! I am your local enterprise knowledge assistant. Ask me anything grounded in your active workspace."
    }


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

    # 1. Dense retrieval (ChromaDB). Stored vectors are unit-normalized, so cosine = 1 - squared_l2 / 2.
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=f"ws_{ws_id}")
        q_vec = get_embedding_model().embed_query(query)
        where_filter = {"clearance": {"$in": authorized_clrs}}
        try:
            results = collection.query(query_embeddings=[q_vec], n_results=5, where=where_filter)
        except Exception:
            # Fallback for unmigrated collections without clearance metadata
            results = collection.query(query_embeddings=[q_vec], n_results=5)
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [2.0] * len(docs)
        for doc, meta, cid, distance in zip(docs, metas, ids, distances):
            meta_dict = meta or {}
            doc_clr = str(meta_dict.get("clearance", "public")).strip().lower()
            if doc_clr not in authorized_clrs:
                continue
            dense_candidates.append({
                "id": cid,
                "content": doc,
                "citation": f"[{cid}] {meta_dict.get('source', 'document')}",
                "similarity": 1.0 - float(distance) / 2.0,
            })
    except NotFoundError:
        degraded.append("not_indexed: this workspace has no vector index yet")
    except Exception as exc:
        degraded.append(f"dense_retrieval_unavailable: {type(exc).__name__}")

    # 2. Sparse retrieval (SQLite FTS5 with BM25 ranking, filtered by clearance, legacy fallback)
    try:
        sparse_candidates = search_sparse_chunks(
            ws_id,
            query,
            top_n=5,
            allowed_clearances=authorized_clrs,
            base_dir=BM25_DIR,
        )
    except Exception as exc:
        degraded.append(f"sparse_retrieval_unavailable: {type(exc).__name__}")

    # 3. Relevance gate
    candidates = dense_candidates + sparse_candidates
    best_similarity = max((c["similarity"] for c in dense_candidates), default=0.0)
    query_terms = _content_terms(query)
    best_coverage = max((_coverage(query_terms, c["content"]) for c in candidates), default=0.0)
    relevant = bool(candidates) and (
        best_similarity >= RELEVANCE_MIN_SIMILARITY or best_coverage >= RELEVANCE_MIN_COVERAGE
    )

    # 4. Fuse with Reciprocal Rank Fusion (RRF) only when the evidence is relevant
    # Fetch top candidates from RRF, then apply Cross-Encoder Reranker for deep attention scoring
    fused_docs = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=6) if relevant else []
    reranked_docs = rerank_documents(query, fused_docs, top_n=3) if fused_docs else []

    chunks_found = [item["content"] for item in reranked_docs]
    citations = [item["citation"] for item in reranked_docs]

    telemetry["retrieval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["dense_candidates_count"] = len(dense_candidates)
    telemetry["sparse_candidates_count"] = len(sparse_candidates)
    telemetry["fused_chunks_count"] = len(chunks_found)
    telemetry["best_similarity"] = round(best_similarity, 4)
    telemetry["best_coverage"] = round(best_coverage, 4)
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

    clean_docs, clean_citations, dropped = sanitize_chunks(raw_docs, raw_citations)

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
    }


def grade_documents_node(state: StateInput) -> dict[str, Any]:
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if not docs:
        if retries >= 2:
            return {"grade_status": "circuit_break"}
        return {"grade_status": "rewrite", "retry_count": retries + 1}

    return {"grade_status": "approved"}


def query_rewrite_node(state: StateInput) -> dict[str, Any]:
    q = state.get("sanitized_question", "")
    return {"sanitized_question": f"{q} details overview"}


def generate_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = state.get("documents", [])
    q = state.get("sanitized_question", "")
    telemetry = dict(state.get("telemetry", {}))
    # Keep retrieval degradations; drop fallback notes from an earlier attempt so they are not repeated.
    degraded = [d for d in state.get("degraded", []) if not d.startswith("generation_fallback")]

    if not docs:
        telemetry["abstained"] = True
        telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {"generation": ABSTAIN_MESSAGE, "telemetry": telemetry, "degraded": degraded}

    context = "\n\n---\n\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely based strictly on the context:"
    telemetry["abstained"] = False
    try:
        response = get_chat_llm().invoke(prompt)
        gen_content = str(response.content).strip()
    except Exception as exc:
        # The model failed. Answer extractively from the same context and flag it for the reviewer.
        gen_content = str(DeterministicContextLLM().invoke(prompt).content).strip()
        degraded.append(f"generation_fallback: {type(exc).__name__}; the answer is extractive, not model-written")
        telemetry["generation_fallback"] = True

    # Real-time outbound DLP filter
    gen_content = redact_pii(gen_content)
    telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "generation": gen_content,
        "telemetry": telemetry,
        "degraded": degraded,
    }


def grade_hallucination_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)
    telemetry = dict(state.get("telemetry", {}))

    status = "grounded"
    update: dict[str, Any] = {}
    ratio: float | None = None

    if docs:
        doc_words = set(_WORD.findall(" ".join(docs).lower()))
        gen_words = [w for w in _WORD.findall(gen.lower()) if len(w) > 3]
        if gen_words:
            ratio = sum(1 for w in gen_words if w in doc_words) / len(gen_words)
            if ratio < 0.25:
                if retries >= 2:
                    status = "circuit_break"
                else:
                    status = "retry"
                    update["retry_count"] = retries + 1

    telemetry["groundedness_ratio"] = None if ratio is None else round(ratio, 4)
    audit_record(
        "groundedness",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "status": status,
            "groundedness_ratio": telemetry["groundedness_ratio"],
            "retry_count": retries,
        },
    )
    return {"hallucination_status": status, "telemetry": telemetry, **update}


def grade_answer_relevance_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    if gen == ABSTAIN_MESSAGE or state.get("telemetry", {}).get("abstained"):
        return {"relevance_status": "relevant"}
    gen_lower = gen.lower()
    gen_words = set(_WORD.findall(gen_lower))
    q = state.get("original_question") or state.get("sanitized_question", "")
    retries = state.get("retry_count", 0)

    q_tokens = [t for t in _WORD.findall(q.lower()) if len(t) > 3]
    if q_tokens and not any(t in gen_words or t in gen_lower for t in q_tokens):
        if retries >= 2:
            return {"relevance_status": "circuit_break"}
        return {"relevance_status": "irrelevant", "retry_count": retries + 1}

    return {"relevance_status": "relevant"}


def _normalize_decision(raw: Any) -> dict[str, str]:
    """Read the resume payload. The approver is the account a valid session token was issued to."""
    payload = raw if isinstance(raw, dict) else {"decision": raw}
    return {
        "decision": str(payload.get("decision", "")).strip().lower(),
        "approver": verify_token(str(payload.get("approver_token", ""))),
        "reason": redact_pii(str(payload.get("reason", "")).strip()),
    }


def human_in_the_loop_node(state: StateInput) -> dict[str, Any]:
    # The payload is what approve.py displays. It carries the redacted question, never the raw input.
    display_q = state.get("original_question") or state.get("sanitized_question", "")
    review = _normalize_decision(interrupt({
        "action": "human_approval",
        "question": display_q,
        "citations": state.get("citations", []),
        "candidate_answer": state.get("generation", ""),
    }))
    gen = state.get("generation", "")
    history = list(state.get("history", []))

    approved = review["decision"] == "approved"
    ws_id = state.get("workspace_id", "default")
    if approved and not review["approver"]:
        # An approval with no accountable reviewer is not released.
        approved = False
        review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: no approver identity"]))
    if approved and review["approver"] == state.get("principal", ""):
        # Separation of duties: the requester cannot approve their own query.
        approved = False
        review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: self-approval"]))
    if approved and not authorize(review["approver"], ws_id, "review"):
        approved = False
        review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: approver lacks review permission"]))

    # Written before the answer is released, so a failed write releases nothing.
    audit_record(
        "hitl_decision",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "decision": "approved" if approved else "rejected",
            "requested": review["decision"],
            "approver": review["approver"],
            "reason": review["reason"],
            "released": approved,
            "answer_sha256": _sha256(gen),
            "citations": state.get("citations", []),
            "groundedness_ratio": state.get("telemetry", {}).get("groundedness_ratio"),
            "hallucination_status": state.get("hallucination_status", ""),
            "relevance_status": state.get("relevance_status", ""),
            "degraded": state.get("degraded", []),
        },
    )

    if approved:
        history.append(f"Assistant: {gen}")

        q = state.get("original_question") or state.get("sanitized_question", "")
        ws_id = state.get("workspace_id", "default")
        principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
        clearance = get_user_clearance(principal) if principal else "public"
        key = cache_key(ws_id, q, clearance)
        SEMANTIC_CACHE[key] = gen
        SEMANTIC_CACHE_CITATIONS[key] = list(state.get("citations", []))

        return {"generation": gen, "history": history}

    rejection_notice = "[RESPONSE REJECTED BY HUMAN COMPLIANCE]"
    history.append(f"Assistant: {rejection_notice}")
    return {"generation": rejection_notice, "citations": [], "degraded": [], "history": history}


# --- GRAPH EDGES & ROUTING ---

def route_safety(state: StateInput) -> str:
    return "pass" if state.get("is_safe", True) else "fail"

def route_cache(state: StateInput) -> str:
    return "hit" if state.get("cache_hit", False) else "miss"

def route_intent(state: StateInput) -> str:
    return state.get("intent", "rag")

def route_doc_grade(state: StateInput) -> str:
    status = state.get("grade_status", "approved")
    if status == "circuit_break":
        return "circuit_break"
    if status == "rewrite":
        return "rewrite"
    return "approved"

def route_hallucination(state: StateInput) -> str:
    status = state.get("hallucination_status", "grounded")
    if status == "circuit_break":
        return "circuit_break"
    if status == "retry":
        return "retry"
    return "grounded"

def route_relevance(state: StateInput) -> str:
    status = state.get("relevance_status", "relevant")
    if status == "circuit_break":
        return "circuit_break"
    if status == "irrelevant":
        return "irrelevant"
    return "relevant"


# --- BUILD STATE GRAPH ---

builder = StateGraph(GraphState)

builder.add_node("guardrail", guardrail_input_node)
builder.add_node("cache_lookup", cache_lookup_node)
builder.add_node("router", router_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("retrieve", hybrid_retrieval_node)
builder.add_node("sanitize_context", sanitize_context_node)
builder.add_node("grade_docs", grade_documents_node)
builder.add_node("rewrite", query_rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("grade_hallucination", grade_hallucination_node)
builder.add_node("grade_relevance", grade_answer_relevance_node)
builder.add_node("human_approval", human_in_the_loop_node)

builder.add_edge(START, "guardrail")

builder.add_conditional_edges(
    "guardrail",
    route_safety,
    {"pass": "cache_lookup", "fail": END}
)

builder.add_conditional_edges(
    "cache_lookup",
    route_cache,
    {"hit": END, "miss": "router"}
)

builder.add_conditional_edges(
    "router",
    route_intent,
    {"chitchat": "chitchat", "rag": "retrieve"}
)

builder.add_edge("chitchat", END)
builder.add_edge("retrieve", "sanitize_context")
builder.add_edge("sanitize_context", "grade_docs")

builder.add_conditional_edges(
    "grade_docs",
    route_doc_grade,
    {"approved": "generate", "rewrite": "rewrite", "circuit_break": "generate"}
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "grade_hallucination")

builder.add_conditional_edges(
    "grade_hallucination",
    route_hallucination,
    {"grounded": "grade_relevance", "retry": "generate", "circuit_break": "human_approval"}
)

builder.add_conditional_edges(
    "grade_relevance",
    route_relevance,
    {"relevant": "human_approval", "irrelevant": "rewrite", "circuit_break": "human_approval"}
)

builder.add_edge("human_approval", END)

workflow = builder
