"""Human-in-the-loop release: auto-approval rules, reviewer checks, audit, and caching of released answers."""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from rag_lab.retrieval.semantic_cache import store_cache
from rag_lab.security.access import authorize
from rag_lab.security.access import clearance_covers
from rag_lab.security.access import get_user_clearance
from rag_lab.security.access import verify_token
from rag_lab.security.audit import record as audit_record
from rag_lab.security.redaction import redact_pii

from rag_lab.lang import detect_language, is_abstain_message, rejection_notice
from rag_lab.pipeline.context import MAX_HISTORY_TURNS
from rag_lab.pipeline.state import ABSTAIN_MESSAGE, BLOCKING_DEGRADED_PREFIXES, StateInput, _sha256


def _normalize_decision(raw: Any) -> dict[str, str]:
    """Read the resume payload. The approver is the account a valid session token was issued to."""
    payload = raw if isinstance(raw, dict) else {"decision": raw}
    return {
        "decision": str(payload.get("decision", "")).strip().lower(),
        "approver": verify_token(str(payload.get("approver_token", ""))),
        "reason": redact_pii(str(payload.get("reason", "")).strip()),
    }


def _review_reasons(state: StateInput) -> list[str]:
    """Why an answer must be looked at by a person even for an account on auto_approve."""
    reasons: list[str] = []
    hallucination = state.get("hallucination_status", "") or "grounded"
    relevance = state.get("relevance_status", "") or "relevant"
    telemetry = state.get("telemetry", {}) or {}
    if hallucination != "grounded":
        reasons.append(f"groundedness check did not pass ({hallucination})")
        # Say why, so the reviewer knows what to look for without reading telemetry.
        if telemetry.get("unsupported_numbers"):
            reasons.append(
                "answer states figures not found in the sources: " + ", ".join(telemetry["unsupported_numbers"])
            )
        if telemetry.get("generation_retry_skipped"):
            reasons.append(f"not regenerated: {telemetry['generation_retry_skipped']}")
    if relevance != "relevant":
        reasons.append(f"relevance check did not pass ({relevance})")
    # An abstention is no exception: "nothing found" from a search that was broken or tampered with is exactly the
    # answer a person must look at. A clean abstention still releases like any other grounded, relevant answer.
    reasons.extend(note for note in state.get("degraded", []) if note.startswith(BLOCKING_DEGRADED_PREFIXES))
    return reasons


def human_in_the_loop_node(state: StateInput) -> dict[str, Any]:
    # The payload is what approve.py displays. It carries the redacted question, never the raw input.
    display_q = state.get("original_question") or state.get("sanitized_question", "")
    gen = state.get("generation", "")
    history = list(state.get("history", []))
    ws_id = state.get("workspace_id", "default")
    principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
    requester_clearance = state.get("principal_clearance") or get_user_clearance(principal)
    abstained = gen == ABSTAIN_MESSAGE or is_abstain_message(gen) or bool(state.get("telemetry", {}).get("abstained"))

    # auto_approve skips the human gate only for answers that passed every automated check.
    reasons = _review_reasons(state)
    has_auto_approve = bool(principal and authorize(principal, ws_id, "auto_approve"))
    auto = has_auto_approve and not reasons
    if auto:
        review = {
            "decision": "approved",
            "approver": f"auto:{principal}",
            "reason": f"auto-approved via policy grant for {principal}",
        }
        approved = True
    else:
        review = _normalize_decision(interrupt({
            "action": "human_approval",
            "question": display_q,
            "asked_question": state.get("asked_question") or display_q,
            "citations": state.get("citations", []),
            "candidate_answer": state.get("generation", ""),
            "requested_by": principal,
            "requester_clearance": requester_clearance,
            "review_reasons": reasons,
        }))
        approved = review["decision"] == "approved"
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
        if approved and not clearance_covers(get_user_clearance(review["approver"]), requester_clearance):
            # A reviewer reads the answer, so they must be cleared for what the requester may see.
            approved = False
            review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: approver clearance is below the requester's"]))

    asked = state.get("asked_question") or display_q

    # Written before the answer is released, so a failed write releases nothing.
    audit_record(
        "hitl_decision",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "question": asked,
            "decision": "approved" if approved else "rejected",
            "requested": review["decision"],
            "approver": review["approver"],
            "reason": review["reason"],
            "released": approved,
            "auto_approved": bool(auto and approved),
            "auto_approve_withheld": bool(has_auto_approve and reasons),
            "review_reasons": reasons,
            "answer_sha256": _sha256(gen),
            "citations": state.get("citations", []),
            "groundedness_ratio": state.get("telemetry", {}).get("groundedness_ratio"),
            "hallucination_status": state.get("hallucination_status", ""),
            "relevance_status": state.get("relevance_status", ""),
            "degraded": state.get("degraded", []),
        },
    )
    if approved:
        # Only released answers become conversation context for later follow-ups.
        history = (history + [{"question": display_q, "asked": asked, "answer": gen, "released": True}])[-MAX_HISTORY_TURNS:]

        q = state.get("original_question") or state.get("sanitized_question", "")
        clearance = get_user_clearance(principal) if principal else "public"
        cites = list(state.get("citations", []))
        # An abstention is not an answer worth remembering: the next question may find the documents.
        if not abstained:
            store_cache(
                workspace_id=ws_id,
                query=q,
                clearance=clearance,
                generation=gen,
                citations=cites,
                reviewed=not auto,
                approved_by=review["approver"],
            )

        return {"generation": gen, "history": history}

    history = (history + [{"question": display_q, "asked": asked, "answer": "", "released": False}])[-MAX_HISTORY_TURNS:]
    user_lang = detect_language(asked or display_q)
    return {"generation": rejection_notice(user_lang), "citations": [], "degraded": [], "history": history}
