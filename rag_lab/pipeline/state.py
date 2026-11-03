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
    history: List[str]
    telemetry: Dict[str, Any]
    degraded: List[str]
    principal_clearance: str
    dropped_chunks: List[Dict[str, Any]]


StateInput = GraphState | dict[str, Any]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ABSTAIN_MESSAGE = "I could not find relevant documentation in this workspace to answer your query."


# Retrieval problems that mean the evidence behind an answer may be incomplete or tampered with.
# An answer carrying one of these is never auto-approved: a person must look at it.
BLOCKING_DEGRADED_PREFIXES = (
    "security_warning",
    "dense_retrieval_unavailable",
    "sparse_retrieval_unavailable",
)
