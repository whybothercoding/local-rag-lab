"""Entry stages: identity and injection guardrail, semantic cache lookup, intent routing, chitchat."""
from __future__ import annotations

from typing import Any
import time

from rag_lab.retrieval.semantic_cache import PersistentCitationsCache
from rag_lab.retrieval.semantic_cache import PersistentSemanticCache
from rag_lab.retrieval.semantic_cache import lookup_cache
from rag_lab.security.access import authorize
from rag_lab.security.access import get_user_clearance
from rag_lab.security.access import verify_token
from rag_lab.security.audit import record as audit_record
from rag_lab.security.context_sanitizer import scan_user_input
from rag_lab.security.redaction import redact_pii

from rag_lab.pipeline.state import StateInput, _sha256
from rag_lab.config import settings
from rag_lab.lang import CHITCHAT_EXAMPLES, CHITCHAT_PHRASES, CHITCHAT_WORDS, chitchat_reply, detect_language, words
from rag_lab.retrieval.embedding_provider import embedding_fingerprint, get_embedding_model
from rag_lab.retrieval.semantic_cache import _cosine_similarity
from rag_lab.pipeline.text import _content_terms


SEMANTIC_CACHE = PersistentSemanticCache()


SEMANTIC_CACHE_CITATIONS = PersistentCitationsCache()


def prepare_pipeline_input(
    *,
    question: str,
    principal_token: str,
    workspace_id: str = "default",
    session_id: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a sanitized initial input payload for the LangGraph pipeline.

    Redacts PII immediately so that LangGraph's initial checkpoint write never
    persists raw PII, regardless of which caller or entry point submits the query.
    """
    redacted = redact_pii(question)
    payload: dict[str, Any] = {
        "workspace_id": workspace_id,
        "session_id": session_id,
        "principal_token": principal_token,
        "question": redacted,
        "original_question": redacted,
        "sanitized_question": "",
        "documents": [],
        "citations": [],
        "generation": "",
        "intent": "",
        "cache_hit": False,
        "is_safe": True,
        "grade_status": "",
        "hallucination_status": "",
        "relevance_status": "",
        "retry_count": 0,
        "generation_retry_count": 0,
        "grader_feedback": "",
        "telemetry": {},
        **kwargs,
    }
    payload["question"] = redacted
    payload["original_question"] = redacted
    return payload


def guardrail_input_node(state: StateInput) -> dict[str, Any]:
    # Callers should pass redacted input (e.g. via prepare_pipeline_input), but this node
    # also enforces redaction and updates state['question'] so unredacted input cannot persist.
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
            "question": denied_query,
            "generation": "[ACCESS DENIED: this account is not authorized for the selected workspace.]",
            "sanitized_question": denied_query,
            "telemetry": telemetry,
        }

    # Same pattern set as retrieved documents, after Unicode and whitespace normalization.
    input_safe, injection_reason = scan_user_input(raw)
    if not input_safe:
        telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        safe_query = redact_pii(raw)
        audit_record(
            "query_received",
            workspace_id=ws_id,
            session_id=session_id,
            data={"status": "blocked", "reason": "prompt_injection", "pattern": injection_reason, "query": safe_query},
        )
        return {
            "is_safe": False,
            "question": safe_query,
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
        "question": sanitized,
        "principal": principal,
        "principal_clearance": get_user_clearance(principal),
        "asked_question": sanitized,
        "original_question": sanitized,
        "sanitized_question": sanitized,
        "telemetry": telemetry,
        # A conversation reuses its thread: every question starts with full retry budgets and no carried-over notes.
        "retry_count": 0,
        "generation_retry_count": 0,
        "grader_feedback": "",
        "degraded": [],
        "dropped_chunks": [],
    }


def cache_lookup_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
    clearance = get_user_clearance(principal) if principal else "public"
    telemetry = dict(state.get("telemetry", {}))

    # A caller who is not on auto_approve is only served answers a person approved.
    may_use_unreviewed = bool(principal and authorize(principal, ws_id, "auto_approve"))
    cache_res = lookup_cache(
        workspace_id=ws_id,
        query=query,
        clearance=clearance,
        include_unreviewed=may_use_unreviewed,
    )
    telemetry["cache_lookup_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if cache_res.get("hit"):
        telemetry["cache_hit"] = True
        telemetry["cache_hit_type"] = cache_res.get("hit_type", "exact")
        telemetry["cache_similarity"] = cache_res.get("similarity", 1.0)
        telemetry["cache_approved_by"] = cache_res.get("approved_by", "")
        cached = cache_res["generation"]
        cached_citations = cache_res.get("citations") or state.get("citations", [])
        audit_record(
            "cache_hit",
            workspace_id=ws_id,
            session_id=state.get("session_id", ""),
            data={
                "query": query,
                "hit_type": telemetry["cache_hit_type"],
                "similarity": telemetry["cache_similarity"],
                "approved_by": telemetry["cache_approved_by"],
                "answer_sha256": _sha256(str(cached)),
            },
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
    tokens = words(state.get("sanitized_question", ""))
    cleaned = " ".join(tokens)
    if any(phrase in cleaned for phrase in CHITCHAT_PHRASES):
        return {"intent": "chitchat"}
    domain_terms = _content_terms(cleaned) - CHITCHAT_WORDS
    if any(w in CHITCHAT_WORDS for w in tokens) and not domain_terms:
        return {"intent": "chitchat"}
    # Greetings and thanks the word lists miss ("How are you doing today?", "Τι κάνεις σήμερα;"): only short
    # questions, and only when they sit as close to a chitchat example as calibrated. A real question routed
    # here would get the canned reply and never reach retrieval, so the threshold leaves a wide margin.
    if len(domain_terms) <= 3 and _chitchat_similarity(state.get("sanitized_question", "")) >= settings.chitchat_min_similarity:
        return {"intent": "chitchat"}
    return {"intent": "rag"}


_EXAMPLE_VECTORS: dict[str, list[list[float]]] = {}


def _chitchat_similarity(question: str) -> float:
    if settings.chitchat_min_similarity > 1 or not question.strip():
        return -1.0
    try:
        model = get_embedding_model()
        key = embedding_fingerprint()
        if key not in _EXAMPLE_VECTORS:
            _EXAMPLE_VECTORS[key] = [model.embed_query(t) for t in CHITCHAT_EXAMPLES]
        vector = model.embed_query(question)
    except Exception:
        return -1.0  # no embedder: questions go to retrieval
    return max(_cosine_similarity(vector, example) for example in _EXAMPLE_VECTORS[key])


def chitchat_node(state: StateInput) -> dict[str, Any]:
    audit_record(
        "chitchat",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={"query": state.get("sanitized_question", "")},
    )
    return {
        "generation": chitchat_reply(detect_language(state.get("asked_question") or state.get("sanitized_question", ""))),
    }
