"""Self-RAG retry loops: separate budgets, grader feedback in the retry prompt, and no futile retries."""
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app import (
    _generation_prompt,
    generate_node,
    grade_answer_relevance_node,
    grade_hallucination_node,
    query_rewrite_node,
    workflow,
)
from config import MAX_GENERATION_RETRIES, MAX_RETRIEVAL_RETRIES
from conftest import token_for
from llm_provider import DeterministicContextLLM

DOCS = ["Rollback SLA is strictly under 5 minutes for production."]
QUESTION = "What is the rollback SLA?"


class ScriptedModel:
    """A stand-in for a real chat model: records prompts and answers from a script."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.answer(prompt, len(self.prompts)))


def _use_model(monkeypatch, model):
    monkeypatch.setattr("app.get_chat_llm", lambda: model)
    monkeypatch.setattr("app.llm_is_fallback", lambda llm: False)


def _real_model_telemetry() -> dict:
    return {"generator_deterministic": False}


# --- feedback reaches the next prompt ------------------------------------------

def test_failed_groundedness_feedback_is_added_to_the_retry_prompt(monkeypatch):
    model = ScriptedModel(lambda prompt, n: "Rollback SLA is strictly under 5 minutes.")
    _use_model(monkeypatch, model)

    first = generate_node({"documents": DOCS, "original_question": QUESTION})
    graded = grade_hallucination_node({
        "documents": DOCS,
        "original_question": QUESTION,
        "generation": "Rollback SLA is strictly under 50 minutes.",
        "telemetry": first["telemetry"],
    })
    assert graded["hallucination_status"] == "retry"
    assert "50" in graded["grader_feedback"]

    second = generate_node({"documents": DOCS, "original_question": QUESTION, **graded})
    assert model.prompts[0] != model.prompts[1], "an identical prompt at temperature 0 repeats the same answer"
    assert "previous answer was rejected because it stated figures that are not in the context (50)" in model.prompts[1]
    assert second["telemetry"]["generation_feedback_applied"] is True
    assert first["telemetry"]["generation_feedback_applied"] is False


def test_feedback_does_not_change_what_the_extractive_synthesizer_reads():
    plain = _generation_prompt("\n\n---\n\n".join(DOCS), QUESTION, "")
    with_feedback = _generation_prompt("\n\n---\n\n".join(DOCS), QUESTION, "it was empty.")
    synth = DeterministicContextLLM()
    assert synth.invoke(plain).content == synth.invoke(with_feedback).content


def test_feedback_is_cleared_when_a_grader_passes_and_when_the_query_is_rewritten():
    passed = grade_hallucination_node({
        "documents": DOCS,
        "generation": "Rollback SLA is strictly under 5 minutes.",
        "grader_feedback": "stale note",
        "telemetry": _real_model_telemetry(),
    })
    assert passed["hallucination_status"] == "grounded"
    assert passed["grader_feedback"] == ""
    assert query_rewrite_node({"original_question": QUESTION, "grader_feedback": "stale"})["grader_feedback"] == ""


def test_generation_answers_the_original_question_not_the_search_rewrite(monkeypatch):
    model = ScriptedModel(lambda prompt, n: "Under 5 minutes.")
    _use_model(monkeypatch, model)
    generate_node({"documents": DOCS, "original_question": QUESTION, "sanitized_question": "rollback sla minutes"})
    assert f"Question: {QUESTION}" in model.prompts[0]


def test_extractive_synthesizer_matches_on_the_search_query(monkeypatch):
    """The synthesizer picks sentences by shared words, so it reads the keyword query that found the documents."""
    seen = []
    original_invoke = DeterministicContextLLM.invoke

    def spy(self, prompt, *args, **kwargs):
        seen.append(prompt)
        return original_invoke(self, prompt, *args, **kwargs)

    monkeypatch.setattr(DeterministicContextLLM, "invoke", spy)
    generate_node({"documents": DOCS, "original_question": QUESTION, "sanitized_question": "rollback sla minutes"})

    class Broken:
        def invoke(self, prompt):
            raise TimeoutError("model hung")

    monkeypatch.setattr("app.get_chat_llm", lambda: Broken())
    generate_node({"documents": DOCS, "original_question": QUESTION, "sanitized_question": "rollback sla minutes"})
    assert len(seen) == 2
    assert all("Question: rollback sla minutes" in prompt for prompt in seen)


# --- separate budgets --------------------------------------------------------------

def test_exhausted_rewrites_do_not_consume_generation_retries():
    result = grade_hallucination_node({
        "documents": DOCS,
        "generation": "Bananas grow abundantly on Mars in solar-powered greenhouses.",
        "retry_count": MAX_RETRIEVAL_RETRIES,
        "generation_retry_count": 0,
        "telemetry": _real_model_telemetry(),
    })
    assert result["hallucination_status"] == "retry"


def test_exhausted_generation_retries_do_not_consume_rewrites(monkeypatch):
    monkeypatch.setattr("app.score_pairs", lambda *a, **k: [-20.0])
    result = grade_answer_relevance_node({
        "original_question": QUESTION,
        "generation": "Expense reports are due monthly.",
        "retry_count": 0,
        "generation_retry_count": MAX_GENERATION_RETRIES,
    })
    assert result["relevance_status"] == "irrelevant"
    assert result["retry_count"] == 1


def test_generation_budget_is_enforced():
    state = {
        "documents": DOCS,
        "generation": "Bananas grow abundantly on Mars in solar-powered greenhouses.",
        "telemetry": _real_model_telemetry(),
    }
    for attempt in range(MAX_GENERATION_RETRIES):
        result = grade_hallucination_node({**state, "generation_retry_count": attempt})
        assert result["hallucination_status"] == "retry"
        assert result["generation_retry_count"] == attempt + 1
    final = grade_hallucination_node({**state, "generation_retry_count": MAX_GENERATION_RETRIES})
    assert final["hallucination_status"] == "circuit_break"


# --- no futile retries for the deterministic generator -------------------------------

def test_deterministic_generator_escalates_without_retrying():
    result = grade_hallucination_node({
        "documents": DOCS,
        "generation": "Rollback SLA is strictly under 50 minutes.",
        "generation_retry_count": 0,
        "telemetry": {"generator_deterministic": True},
    })
    assert result["hallucination_status"] == "circuit_break"
    assert "generation_retry_count" not in result
    assert "deterministic generator" in result["telemetry"]["generation_retry_skipped"]


def test_generate_marks_deterministic_and_fallback_generators(monkeypatch):
    det = generate_node({"documents": DOCS, "original_question": QUESTION})
    assert det["telemetry"]["generator_deterministic"] is True  # tests run LLM_PROVIDER=deterministic

    _use_model(monkeypatch, ScriptedModel(lambda prompt, n: "Under 5 minutes."))
    real = generate_node({"documents": DOCS, "original_question": QUESTION})
    assert real["telemetry"]["generator_deterministic"] is False

    class Broken:
        def invoke(self, prompt):
            raise TimeoutError("model hung")

    monkeypatch.setattr("app.get_chat_llm", lambda: Broken())
    failed = generate_node({"documents": DOCS, "original_question": QUESTION})
    assert failed["telemetry"]["generator_deterministic"] is True
    assert failed["telemetry"]["generation_fallback"] is True


# --- the compiled graph --------------------------------------------------------------

def _run(engine, thread: str) -> tuple[dict, tuple, int]:
    config = {"configurable": {"thread_id": thread}}
    inputs = {
        "workspace_id": "default",
        "session_id": thread,
        "principal_token": token_for("carol"),
        "question": QUESTION,
        "telemetry": {},
        "degraded": [],
    }
    steps = sum(1 for _ in engine.stream(inputs, config=config))
    state = engine.get_state(config)
    return state.values, state.next, steps


def test_graph_recovers_from_a_fabricated_figure_using_feedback(rollback_index, monkeypatch):
    def answer(prompt, n):
        return "Rollback SLA is strictly under 5 minutes." if "rejected because" in prompt else "Rollback SLA is under 50 minutes."

    model = ScriptedModel(answer)
    _use_model(monkeypatch, model)
    engine = workflow.compile(checkpointer=MemorySaver())
    values, pending, _ = _run(engine, "retry-recovers")

    assert pending == ("human_approval",)
    assert values["generation"] == "Rollback SLA is strictly under 5 minutes."
    assert values["hallucination_status"] == "grounded"
    assert values["generation_retry_count"] == 1
    assert len([p for p in model.prompts if p.startswith("Context:")]) == 2


def test_worst_case_path_terminates_within_both_budgets(rollback_index, monkeypatch):
    """Every grader fails every time. The run must still end at review, having spent exactly both budgets."""
    monkeypatch.setattr("app.score_pairs", lambda *a, **k: [-20.0])

    def answer(prompt, n):
        # Fabricated figure on first try (groundedness fails), then grounded but off-topic (relevance fails).
        return "Strictly under 50 minutes." if "rejected because" not in prompt and n == 1 else "Strictly under minutes."

    model = ScriptedModel(answer)
    _use_model(monkeypatch, model)
    engine = workflow.compile(checkpointer=MemorySaver())
    values, pending, steps = _run(engine, "worst-case")

    assert pending == ("human_approval",)
    assert values["relevance_status"] == "circuit_break"
    assert values["retry_count"] == MAX_RETRIEVAL_RETRIES
    assert values["generation_retry_count"] == MAX_GENERATION_RETRIES
    # One answer per retrieval round plus one per generation retry: bounded, never a loop.
    # (The first rewrite also asks the model for a keyword query; that call is not an answer.)
    answers = [p for p in model.prompts if p.startswith("Context:")]
    assert len(answers) == (MAX_RETRIEVAL_RETRIES + 1) + MAX_GENERATION_RETRIES
    # Exact step count: 3 entry nodes, 6 nodes per retrieval round, a rewrite per extra round, 2 per
    # generation retry, and the review node. LangGraph >= 1.1 (the declared floor) allows 10,000 by default;
    # 1.0 allowed 25, which this path (26 at default budgets) would exceed.
    expected = 3 + 6 * (MAX_RETRIEVAL_RETRIES + 1) + MAX_RETRIEVAL_RETRIES + 2 * MAX_GENERATION_RETRIES + 1
    assert steps == expected, steps


def test_deterministic_fallback_goes_straight_to_review(rollback_index, monkeypatch):
    """With the extractive synthesizer a failed groundedness check escalates instead of looping."""
    monkeypatch.setattr("app._unsupported_numbers", lambda *a, **k: {"99"})
    engine = workflow.compile(checkpointer=MemorySaver())
    values, pending, _ = _run(engine, "deterministic-escalates")

    assert pending == ("human_approval",)
    assert values["hallucination_status"] == "circuit_break"
    assert values.get("generation_retry_count", 0) == 0
    assert "deterministic generator" in values["telemetry"]["generation_retry_skipped"]

    # The reviewer is told why, in the payload the review screens display.
    state = engine.get_state({"configurable": {"thread_id": "deterministic-escalates"}})
    reasons = state.tasks[0].interrupts[0].value["review_reasons"]
    assert "groundedness check did not pass (circuit_break)" in reasons
    assert "answer states figures not found in the sources: 99" in reasons
    assert any(r.startswith("not regenerated: deterministic generator") for r in reasons)


def test_a_new_question_in_the_same_thread_starts_with_fresh_budgets(rollback_index, monkeypatch):
    """The UI keeps one thread per conversation; spent budgets and stale feedback must not carry over."""
    from app import guardrail_input_node

    reset = guardrail_input_node({
        "question": QUESTION,
        "principal_token": token_for("carol"),
        "workspace_id": "default",
        "retry_count": MAX_RETRIEVAL_RETRIES,
        "generation_retry_count": MAX_GENERATION_RETRIES,
        "grader_feedback": "stale",
        "degraded": ["security_warning: from the previous question"],
    })
    assert reset["retry_count"] == 0
    assert reset["generation_retry_count"] == 0
    assert reset["grader_feedback"] == ""
    assert reset["degraded"] == []

    def answer(prompt, n):
        return "Rollback SLA is strictly under 5 minutes." if "rejected because" in prompt else "Under 50 minutes."

    model = ScriptedModel(answer)
    _use_model(monkeypatch, model)
    engine = workflow.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "conversation"}}
    first, *_ = _run(engine, "conversation")
    assert first["generation_retry_count"] == 1
    engine.update_state(config, {"generation": first["generation"]}, as_node="human_approval")

    second, pending, _ = _run(engine, "conversation")
    assert pending == ("human_approval",)
    assert second["hallucination_status"] == "grounded", "the second question still had its generation retry"
