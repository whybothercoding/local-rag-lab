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
from rag_lab.lang import CHITCHAT_PHRASES, CHITCHAT_WORDS, words
from rag_lab.pipeline.text import _content_terms


SEMANTIC_CACHE = PersistentSemanticCache()


SEMANTIC_CACHE_CITATIONS = PersistentCitationsCache()


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
        "principal_clearance": get_user_clearance(principal),
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
