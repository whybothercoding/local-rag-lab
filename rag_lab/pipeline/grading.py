"""Self-RAG graders: groundedness (numeric, lexical, cross-encoder) and answer relevance."""
from __future__ import annotations

from typing import Any

from rag_lab.config import settings
from rag_lab.retrieval.reranker import score_pairs
from rag_lab.security.audit import record as audit_record

from rag_lab.pipeline.state import ABSTAIN_MESSAGE, StateInput
import re

from rag_lab.lang import fold, is_abstain_message, is_no_info_sentence, strip_attribution, words
from rag_lab.pipeline.text import STOPWORDS, _unsupported_numbers, split_sentences


_FEEDBACK_UNSUPPORTED = (
    "it was not supported by the context. State only what the context says, or say that the context "
    "does not contain the answer."
)


_FEEDBACK_EMPTY = "it was empty. Answer from the context, or say that the context does not contain the answer."


def _generation_failure(state: StateInput, telemetry: dict[str, Any], feedback: str) -> tuple[str, dict[str, Any]]:
    """Decide what a failed groundedness check does next: regenerate with feedback, or stop and escalate.

    A deterministic generator would return the same text, so it escalates at once instead of looping.
    """
    retries = int(state.get("generation_retry_count", 0))
    if telemetry.get("generator_deterministic"):
        telemetry["generation_retry_skipped"] = "deterministic generator; a retry would repeat the same answer"
        return "circuit_break", {}
    if retries >= settings.max_generation_retries:
        return "circuit_break", {}
    return "retry", {"generation_retry_count": retries + 1, "grader_feedback": feedback}


_EXTRACTIVE_PREFIX = re.compile(r"^Based on available documentation:\s*")


def _lexical_ratio(gen: str, docs: list[str]) -> float | None:
    doc_words = set(words(" ".join(docs)))
    gen_words = [w for w in words(gen) if len(w) > 3]
    return sum(1 for w in gen_words if w in doc_words) / len(gen_words) if gen_words else None


def _nli_verdict(gen: str, docs: list[str], telemetry: dict[str, Any]) -> list[str] | None:
    """Unsupported answer sentences ([] when every claim is entailed), or None when NLI is unavailable."""
    from rag_lab.retrieval.nli import NLIUnavailableError, sentence_support

    claims = [strip_attribution(s) for s in split_sentences(_EXTRACTIVE_PREFIX.sub("", gen)) if not is_no_info_sentence(s)]
    try:
        scores = [(sentence_support(s, docs), s) for s in claims]
    except NLIUnavailableError:
        return None
    telemetry["nli_min_support"] = round(min((v for v, _ in scores), default=1.0), 4)
    telemetry["nli_claims"] = len(scores)
    unsupported = [s for v, s in scores if v < settings.groundedness_min_entailment]
    telemetry["nli_supported"] = not unsupported
    return unsupported


def grade_hallucination_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    docs = state.get("documents", [])
    telemetry = dict(state.get("telemetry", {}))
    telemetry.pop("generation_retry_skipped", None)

    status = "grounded"
    update: dict[str, Any] = {"grader_feedback": ""}
    ratio: float | None = None
    semantic_score: float | None = None
    grader_method = "lexical_pass"

    if gen == ABSTAIN_MESSAGE or is_abstain_message(gen) or telemetry.get("abstained"):
        status = "grounded"
        grader_method = "abstain"
    elif not gen or not str(gen).strip():
        # Blank generation cannot be grounded
        grader_method = "empty_generation"
        status, update = _generation_failure(state, telemetry, _FEEDBACK_EMPTY)
    elif docs and (
        unsupported := _unsupported_numbers(
            gen, docs, state.get("original_question") or state.get("sanitized_question", "")
        )
    ):
        # A figure the sources do not contain is never grounded, however much other wording overlaps.
        grader_method = "numeric_mismatch"
        telemetry["unsupported_numbers"] = sorted(unsupported)
        feedback = (
            f"it stated figures that are not in the context ({', '.join(sorted(unsupported))}). "
            "Use only figures that appear in the context."
        )
        status, update = _generation_failure(state, telemetry, feedback)
    elif docs and settings.groundedness_method == "nli" and (verdict := _nli_verdict(gen, docs, telemetry)) is not None:
        # Every claim must be stated by a source passage; the word-overlap ratio is kept as telemetry.
        grader_method = "nli"
        ratio = _lexical_ratio(gen, docs)
        if verdict:
            sentence = verdict[0]
            telemetry["unsupported_sentence"] = sentence[:200]
            feedback = (f'it stated something the context does not support: "{sentence[:160]}". '
                        "State only what the context says.")
            grader_method = "nli_rejected"
            status, update = _generation_failure(state, telemetry, feedback)
    elif docs:
        if settings.groundedness_method == "nli":
            telemetry["groundedness_fallback"] = "NLI model unavailable; lexical check used"
        doc_words = set(words(" ".join(docs)))
        gen_words = [w for w in words(gen) if len(w) > 3]
        if gen_words:
            ratio = sum(1 for w in gen_words if w in doc_words) / len(gen_words)
            if ratio < settings.groundedness_min_ratio:
                # Fast lexical screening missed (<25% overlap). Fall back to pinned
                # in-process ONNX cross-encoder to detect semantically grounded paraphrases.
                scores = score_pairs(gen, docs[:5])
                if scores:
                    semantic_score = round(max(scores), 4)
                    if semantic_score >= settings.groundedness_cross_encoder_threshold:
                        grader_method = "semantic_cross_encoder"
                    else:
                        grader_method = "semantic_rejected"
                        status, update = _generation_failure(state, telemetry, _FEEDBACK_UNSUPPORTED)
                else:
                    grader_method = "lexical_rejected"
                    status, update = _generation_failure(state, telemetry, _FEEDBACK_UNSUPPORTED)

    telemetry["groundedness_ratio"] = None if ratio is None else round(ratio, 4)
    telemetry["groundedness_grader"] = grader_method
    if semantic_score is not None:
        telemetry["semantic_groundedness_score"] = semantic_score

    audit_record(
        "groundedness",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "status": status,
            "grader": grader_method,
            "groundedness_ratio": telemetry["groundedness_ratio"],
            "semantic_score": semantic_score,
            "generation_retry_count": int(state.get("generation_retry_count", 0)),
            "retry_skipped": telemetry.get("generation_retry_skipped"),
        },
    )
    return {"hallucination_status": status, "telemetry": telemetry, **update}


def grade_answer_relevance_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    telemetry = dict(state.get("telemetry", {}))
    if gen == ABSTAIN_MESSAGE or is_abstain_message(gen) or telemetry.get("abstained"):
        telemetry["relevance_grader"] = "abstain"
        return {"relevance_status": "relevant", "telemetry": telemetry}

    if not gen or not str(gen).strip():
        # Blank generation cannot be a relevant answer
        telemetry["relevance_grader"] = "empty_generation"
        if state.get("retry_count", 0) >= settings.max_retrieval_retries:
            return {"relevance_status": "circuit_break", "telemetry": telemetry}
        return {"relevance_status": "irrelevant", "retry_count": state.get("retry_count", 0) + 1, "telemetry": telemetry}

    gen_lower = fold(gen)
    gen_words = set(words(gen))
    q = state.get("original_question") or state.get("sanitized_question", "")
    retries = state.get("retry_count", 0)

    q_tokens = [t for t in words(q) if len(t) > 3 and t not in STOPWORDS]
    has_lexical_overlap = bool(not q_tokens or any(t in gen_words or t in gen_lower for t in q_tokens))

    status = "relevant"
    grader_method = "lexical_pass"
    semantic_score: float | None = None
    update: dict[str, Any] = {}

    if not has_lexical_overlap:
        # Fast lexical check missed (>3 char query tokens absent). Fall back to
        # pinned in-process ONNX cross-encoder to assess semantic relevance.
        scores = score_pairs(q, [gen])
        if scores:
            semantic_score = round(scores[0], 4)
            if semantic_score >= settings.relevance_cross_encoder_threshold:
                grader_method = "semantic_cross_encoder"
            else:
                grader_method = "semantic_rejected"
                if retries >= settings.max_retrieval_retries:
                    status = "circuit_break"
                else:
                    status = "irrelevant"
                    update["retry_count"] = retries + 1
        else:
            grader_method = "lexical_rejected"
            if retries >= settings.max_retrieval_retries:
                status = "circuit_break"
            else:
                status = "irrelevant"
                update["retry_count"] = retries + 1

    telemetry["relevance_grader"] = grader_method
    if semantic_score is not None:
        telemetry["semantic_relevance_score"] = semantic_score

    audit_record(
        "answer_relevance",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "status": status,
            "grader": grader_method,
            "semantic_score": semantic_score,
            "retry_count": retries,
        },
    )
    return {"relevance_status": status, "telemetry": telemetry, **update}
