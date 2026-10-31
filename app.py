import os
import re
import time
import hashlib
from typing import TypedDict, List, Dict, Any, Optional

import chromadb

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from config import (
    DB_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBED_MODEL,
    CHROMA_PATH,
    BM25_DIR,
)
from embedding_provider import get_embedding_model
from llm_provider import get_chat_llm
from sparse_search import search_sparse_chunks


class GraphState(TypedDict, total=False):
    workspace_id: str
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


StateInput = GraphState | dict[str, Any]
SEMANTIC_CACHE: dict[str, str] = {}


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
    t0 = time.perf_counter()
    raw = state.get("question", "")
    telemetry = dict(state.get("telemetry", {}))

    adversarial_patterns = [
        r"ignore previous instructions",
        r"system prompt override",
        r"bypass security rules",
        r"disregard all prior directives",
    ]
    for pattern in adversarial_patterns:
        if re.search(pattern, raw, re.IGNORECASE):
            telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "is_safe": False,
                "generation": "[SECURITY VIOLATION: Adversarial prompt injection detected.]",
                "sanitized_question": raw,
                "telemetry": telemetry,
            }

    # PII sanitization: email and phone redaction
    sanitized = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "[REDACTED_EMAIL]",
        raw,
    )
    sanitized = re.sub(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", "[REDACTED_PHONE]", sanitized)

    telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "is_safe": True,
        "sanitized_question": sanitized,
        "telemetry": telemetry,
    }


def cache_lookup_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    telemetry = dict(state.get("telemetry", {}))
    key = hashlib.sha256(f"{ws_id}:{query.strip().lower()}".encode()).hexdigest()

    telemetry["cache_lookup_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if key in SEMANTIC_CACHE:
        telemetry["cache_hit"] = True
        return {
            "cache_hit": True,
            "generation": SEMANTIC_CACHE[key],
            "telemetry": telemetry,
        }

    telemetry["cache_hit"] = False
    return {"cache_hit": False, "telemetry": telemetry}


def router_node(state: StateInput) -> dict[str, Any]:
    query = state.get("sanitized_question", "").lower()
    chitchat_keywords = ["hello", "hi", "hey", "who are you", "what can you do", "thanks", "thank you"]

    if any(k in query.split() for k in chitchat_keywords):
        return {"intent": "chitchat"}
    return {"intent": "rag"}


def chitchat_node(state: StateInput) -> dict[str, Any]:
    return {
        "generation": "Hello! I am your local enterprise knowledge assistant. Ask me anything grounded in your active workspace."
    }


def hybrid_retrieval_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    telemetry = dict(state.get("telemetry", {}))

    dense_candidates: List[Dict[str, Any]] = []
    sparse_candidates: List[Dict[str, Any]] = []

    # 1. Dense retrieval (ChromaDB)
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=f"ws_{ws_id}")
        embeddings = get_embedding_model()
        q_vec = embeddings.embed_query(query)

        results = collection.query(query_embeddings=[q_vec], n_results=5)
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else []
            ids = results["ids"][0] if "ids" in results and results["ids"] else []
            for doc, meta, cid in zip(docs, metas, ids):
                src = meta.get("source", "document")
                dense_candidates.append({
                    "id": cid,
                    "content": doc,
                    "citation": f"[{cid}] {src}"
                })
    except Exception:
        pass

    # 2. Sparse retrieval (Safe SQLite FTS5 with BM25 ranking & legacy fallback)
    try:
        sparse_candidates = search_sparse_chunks(ws_id, query, top_n=5, base_dir=BM25_DIR)
    except Exception:
        sparse_candidates = []

    # 3. Fuse with Reciprocal Rank Fusion (RRF)
    fused_docs = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=3)

    chunks_found = [item["content"] for item in fused_docs]
    citations = [item["citation"] for item in fused_docs]

    # Preserve pre-seeded test or passed documents if retrieval found no candidates
    if not chunks_found and state.get("documents"):
        chunks_found = list(state.get("documents", []))
        citations = list(state.get("citations", []))

    telemetry["retrieval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["dense_candidates_count"] = len(dense_candidates)
    telemetry["sparse_candidates_count"] = len(sparse_candidates)
    telemetry["fused_chunks_count"] = len(chunks_found)

    return {
        "documents": chunks_found,
        "citations": citations,
        "telemetry": telemetry
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

    if not docs:
        telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "generation": "I could not find relevant documentation in this workspace to answer your query.",
            "telemetry": telemetry
        }

    context = "\n\n---\n\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely based strictly on the context:"
    try:
        llm = get_chat_llm()
        response = llm.invoke(prompt)
        gen_content = str(response.content).strip()
    except Exception:
        # Fallback if local model execution fails
        gen_content = f"Based on the documentation: {docs[0]}"

    telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "generation": gen_content,
        "telemetry": telemetry
    }


def grade_hallucination_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if not docs:
        return {"hallucination_status": "grounded"}

    doc_words = set(" ".join(docs).lower().split())
    gen_words = [w for w in gen.lower().split() if len(w) > 3]

    if gen_words:
        grounded_ratio = sum(1 for w in gen_words if w in doc_words) / len(gen_words)
        if grounded_ratio < 0.25:
            if retries >= 2:
                return {"hallucination_status": "circuit_break"}
            return {"hallucination_status": "retry", "retry_count": retries + 1}

    return {"hallucination_status": "grounded"}


def grade_answer_relevance_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "").lower()
    q = state.get("sanitized_question", "").lower()
    retries = state.get("retry_count", 0)

    q_tokens = [t for t in q.split() if len(t) > 3]
    if q_tokens and not any(t in gen for t in q_tokens):
        if retries >= 2:
            return {"relevance_status": "circuit_break"}
        return {"relevance_status": "irrelevant", "retry_count": retries + 1}

    return {"relevance_status": "relevant"}


def human_in_the_loop_node(state: StateInput) -> dict[str, Any]:
    decision = interrupt("human_approval")
    history = list(state.get("history", []))

    if decision == "approved":
        gen = state.get("generation", "")
        history.append(f"Assistant: {gen}")

        q = state.get("sanitized_question", "")
        ws_id = state.get("workspace_id", "default")
        key = hashlib.sha256(f"{ws_id}:{q.strip().lower()}".encode()).hexdigest()
        SEMANTIC_CACHE[key] = gen

        return {"generation": gen, "history": history}
    else:
        rejection_notice = "[RESPONSE REJECTED BY HUMAN COMPLIANCE]"
        history.append(f"Assistant: {rejection_notice}")
        return {"generation": rejection_notice, "history": history}


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
builder.add_edge("retrieve", "grade_docs")

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
