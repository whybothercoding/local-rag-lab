"""Follow-up questions: rewrite "And for longer flights?" into a standalone question before anything uses it.

Runs after the guardrail and before the cache, so the cache key, retrieval, generation, and relevance
grading all see a question that means the same thing outside this conversation. Only answers that were
released enter the history (see review.py), so an unapproved answer never shapes a later question.

The rewrite is new input: it is scanned for injection and masked for PII again, and an unsafe or empty
rewrite falls back to the literal question. The user's literal words stay in `asked_question` for the
reviewer, the audit trail, and the UI.
"""
from __future__ import annotations

import re
from typing import Any

from rag_lab.lang import CHITCHAT_WORDS, STOPWORDS, fold, words
from rag_lab.pipeline import llm_provider
from rag_lab.pipeline.llm_provider import DeterministicContextLLM
from rag_lab.pipeline.state import StateInput
from rag_lab.pipeline.text import _content_terms, _keywords
from rag_lab.security.audit import record as audit_record
from rag_lab.security.context_sanitizer import scan_user_input
from rag_lab.security.redaction import redact_pii

# Turns kept in state and shown to the rewriter.
MAX_HISTORY_TURNS = 6
CONTEXT_TURNS = 3

# Openers and pronouns that make a question lean on the previous turn, in every supported language.
_CONTINUATIONS = {fold(w) for w in (
    "and", "also", "what about", "how about", "what if", "und", "auch", "was ist mit", "und wenn",
    "et", "aussi", "et pour", "et si", "και", "επίσης", "και για", "τι γίνεται με",
)}
# Only words that cannot also be an article or a relative word ("la", "les", "τους", "that" are both).
_ANAPHORA = {fold(w) for w in (
    "it", "they", "them", "there", "those", "then",
    "dafür", "danach", "dabei", "davon", "darüber", "dort",
    "ça", "cela", "ceci",
    "αυτό", "αυτά", "εκεί", "γι",
)}
_REWRITE_PROMPT = (
    "Rewrite the last question so that it can be understood without the conversation, in the same "
    "language as the question. Keep its meaning; add only what the earlier turns make clear. If it is "
    "already self-contained, return it unchanged. Reply with the question only.\n\n"
    "Conversation:\n{turns}\n\nLast question: {question}\nStandalone question:"
)


def released_turns(history: list[Any]) -> list[dict[str, str]]:
    """Earlier question/answer pairs that were released. Legacy plain-string entries are ignored."""
    return [h for h in history if isinstance(h, dict) and h.get("released") and h.get("question")]


def looks_like_follow_up(question: str) -> bool:
    folded = " ".join(words(question))
    if any(folded == c or folded.startswith(c + " ") for c in _CONTINUATIONS):
        return True
    if set(folded.split()) & _ANAPHORA:
        return True
    return len(_content_terms(question) - CHITCHAT_WORDS) < 2


def _heuristic_rewrite(question: str, previous: str) -> str:
    """Add the previous question's key words that the follow-up does not already carry."""
    present = set(words(question))
    carried = [w for w in _keywords(previous) if w not in present and w not in STOPWORDS]
    return f"{question} ({' '.join(carried)})" if carried else question


def _model_rewrite(question: str, turns: list[dict[str, str]]) -> str | None:
    try:
        llm = llm_provider.get_chat_llm()
        if llm_provider.llm_is_fallback(llm) or isinstance(llm, DeterministicContextLLM):
            return None
        lines = "\n".join(f"Q: {t['question']}\nA: {t.get('answer', '')}" for t in turns)
        reply = str(llm.invoke(_REWRITE_PROMPT.format(turns=lines, question=question)).content)
        reply = re.sub(r"(?is)<think>.*?(</think>|\Z)", "", reply)
        candidate = " ".join(reply.split()).strip().strip('"').strip()
        return candidate[:500] or None
    except Exception:
        return None


def contextualize_node(state: StateInput) -> dict[str, Any]:
    question = state.get("original_question") or state.get("sanitized_question", "")
    turns = released_turns(list(state.get("history", [])))[-CONTEXT_TURNS:]
    telemetry = dict(state.get("telemetry", {}))
    if not turns or not looks_like_follow_up(question):
        telemetry["contextualized"] = False
        return {"telemetry": telemetry}

    rewritten = _model_rewrite(question, turns)
    method = "model"
    if not rewritten:
        rewritten, method = _heuristic_rewrite(question, turns[-1]["question"]), "keywords"
    # The rewrite is new input: it may carry text from earlier answers.
    rewritten = redact_pii(rewritten)
    safe, reason = scan_user_input(rewritten)
    if not safe:
        rewritten, method = question, f"rejected:{reason}"

    telemetry.update({"contextualized": rewritten != question, "contextualize_method": method})
    if rewritten != question:
        audit_record(
            "question_contextualized",
            workspace_id=state.get("workspace_id", "default"),
            session_id=state.get("session_id", ""),
            data={"asked": question, "standalone": rewritten, "method": method},
        )
    return {"original_question": rewritten, "sanitized_question": rewritten, "telemetry": telemetry}
