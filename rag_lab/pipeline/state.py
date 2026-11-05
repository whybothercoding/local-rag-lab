"""Graph state schema and the constants every pipeline stage shares."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict
import hashlib


class GraphState(TypedDict, total=False):
    workspace_id: str
    session_id: str
    principal_token: str
    principal: str
    question: str
    # The user's literal (redacted) question, for the reviewer, the audit trail, and the UI.
    asked_question: str
    # The question in standalone form: a follow-up rewritten with its conversation (context.py).
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
    # Retrieval retries (query rewrites). The name predates the split and is kept so saved threads still load.
    retry_count: int
    generation_retry_count: int
    # Why the last answer failed a grader, added to the next generation prompt. Fixed text plus extracted figures only.
    grader_feedback: str
    # Released turns: {"question", "answer", "released"}; older threads may hold plain strings.
    history: List[Any]
    telemetry: Dict[str, Any]
    degraded: List[str]
    principal_clearance: str
    dropped_chunks: List[Dict[str, Any]]


# Node input. Tests and callers pass plain dicts; GraphState is total=False, so every key is optional.
StateInput = GraphState


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ABSTAIN_MESSAGE = "I could not find relevant documentation in this workspace to answer your query."

# What a rejected answer is replaced by: the saved answer and every screen that shows it use this one text.
REJECTION_NOTICE = "[RESPONSE REJECTED BY HUMAN COMPLIANCE]"


# Retrieval problems that mean the evidence behind an answer may be incomplete or tampered with.
# An answer carrying one of these is never auto-approved: a person must look at it.
BLOCKING_DEGRADED_PREFIXES = (
    "security_warning",
    "dense_retrieval_unavailable",
    "sparse_retrieval_unavailable",
)
