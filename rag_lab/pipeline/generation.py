"""Answer generation from retrieved context, with grader feedback on retries and a labelled fallback."""
from __future__ import annotations

from typing import Any
import time

from rag_lab.pipeline import llm_provider
from rag_lab.pipeline.llm_provider import DeterministicContextLLM
from rag_lab.security.redaction import redact_pii

from rag_lab.pipeline.state import ABSTAIN_MESSAGE, StateInput


def _generation_prompt(context: str, question: str, feedback: str) -> str:
    """The answer prompt. Grader feedback goes after the instruction, so the extractive synthesizer, which
    parses the Context and Question sections, reads the same question either way."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based strictly on the context:"
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback} Write a corrected answer."
    return prompt


def generate_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = state.get("documents", [])
    # A model answers the user's own question. The extractive synthesizer matches sentences by shared words,
    # so it is given the search query, which after a rewrite is the keyword form that found these documents.
    original_q = state.get("original_question") or state.get("sanitized_question", "")
    search_q = state.get("sanitized_question") or original_q
    telemetry = dict(state.get("telemetry", {}))
    # Keep retrieval degradations; drop fallback notes from an earlier attempt so they are not repeated.
    degraded = [d for d in state.get("degraded", []) if not d.startswith("generation_fallback")]

    if not docs:
        telemetry["abstained"] = True
        telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {"generation": ABSTAIN_MESSAGE, "telemetry": telemetry, "degraded": degraded}

    context = "\n\n---\n\n".join(docs)
    feedback = str(state.get("grader_feedback") or "")
    telemetry["abstained"] = False
    telemetry["generation_feedback_applied"] = bool(feedback)
    prompt = _generation_prompt(context, search_q, feedback)
    try:
        llm = llm_provider.get_chat_llm()
        if llm_provider.llm_is_fallback(llm):
            degraded.append("generation_fallback: the model server is unavailable; the answer is extractive, not model-written")
            telemetry["generation_fallback"] = True
        # The extractive synthesizer returns the same text for the same documents, so retrying it is pointless.
        deterministic = isinstance(llm, DeterministicContextLLM)
        telemetry["generator_deterministic"] = deterministic
        if not deterministic:
            prompt = _generation_prompt(context, original_q, feedback)
        response = llm.invoke(prompt)
        gen_content = str(response.content).strip()
    except Exception as exc:
        # The model failed. Answer extractively from the same context and flag it for the reviewer.
        gen_content = str(
            DeterministicContextLLM().invoke(_generation_prompt(context, search_q, feedback)).content
        ).strip()
        degraded.append(f"generation_fallback: {type(exc).__name__}; the answer is extractive, not model-written")
        telemetry["generation_fallback"] = True
        telemetry["generator_deterministic"] = True

    # Real-time outbound DLP filter
    gen_content = redact_pii(gen_content)
    telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "generation": gen_content,
        "telemetry": telemetry,
        "degraded": degraded,
    }
