"""Test caller boundary PII redaction and initial state protection."""
from langgraph.checkpoint.memory import MemorySaver

from rag_lab.pipeline.entry import guardrail_input_node, prepare_pipeline_input
from rag_lab.pipeline.graph import workflow
from rag_lab.security.access import issue_token


def test_prepare_pipeline_input_redacts_pii():
    raw = "My phone is 555-123-4567 and email is sensitive@corp.com"
    inputs = prepare_pipeline_input(
        workspace_id="default",
        principal_token=issue_token("alice"),
        question=raw,
    )
    assert "sensitive@corp.com" not in inputs["question"]
    assert "555-123-4567" not in inputs["question"]
    assert "[REDACTED_EMAIL]" in inputs["question"]
    assert "[REDACTED_PHONE]" in inputs["question"]
    assert inputs["question"] == inputs["original_question"]


def test_guardrail_input_node_updates_question_in_state():
    """Even if an untrusted caller sends unredacted input directly to the node, the returned state updates question."""
    raw = "Call 555-123-4567 or email bob@example.com"
    state = {
        "workspace_id": "default",
        "principal_token": issue_token("alice"),
        "question": raw,
        "telemetry": {},
    }
    result = guardrail_input_node(state)
    assert result["is_safe"] is True
    assert "bob@example.com" not in result["question"]
    assert "555-123-4567" not in result["question"]
    assert "[REDACTED_EMAIL]" in result["question"]
    assert "[REDACTED_PHONE]" in result["question"]


def test_initial_checkpoint_stores_redacted_pii():
    """Verifies that running the compiled graph with prepare_pipeline_input stores redacted PII in the checkpoint."""
    saver = MemorySaver()
    app = workflow.compile(checkpointer=saver)
    raw = "Secret inquiry for alice@corp.com"
    inputs = prepare_pipeline_input(
        workspace_id="default",
        session_id="thread-test-pii",
        principal_token=issue_token("alice"),
        question=raw,
    )

    config = {"configurable": {"thread_id": "thread-test-pii"}}
    for _ in app.stream(inputs, config=config):
        pass

    state = app.get_state(config)
    assert "alice@corp.com" not in state.values.get("question", "")
    assert "[REDACTED_EMAIL]" in state.values.get("question", "")
