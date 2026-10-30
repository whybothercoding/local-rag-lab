import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock
from psycopg_pool import ConnectionPool

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    workflow,
    DB_URI,
    SEMANTIC_CACHE,
    GradeDocuments,
    GradeHallucinations,
    GradeAnswerRelevance,
    InjectionCheck,
)
from approve import (
    list_pending_threads,
    inspect_thread,
    process_decision,
)
from langgraph.checkpoint.postgres import PostgresSaver


@pytest.fixture(scope="module")
def postgres_pool():
    """Provides a ConnectionPool to the local PostgreSQL container.
    Skips the integration suite if Postgres is not reachable.
    """
    try:
        # tests/test_integration.py (inside postgres_pool fixture)
        pool = ConnectionPool(conninfo=DB_URI, max_size=5, open=True, kwargs={"autocommit": True})
        # Test connection health
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as exc:
        pytest.skip(f"Local PostgreSQL container not reachable at {DB_URI}: {exc}")

    yield pool
    pool.close()


def test_full_pipeline_to_postgres_and_approve_cli(postgres_pool, mocker):
    """End-to-end integration test:
    1. app.py runs up to the interrupt and saves to Postgres.
    2. approve.py discovers the pending thread in Postgres.
    3. approve.py inspects and resumes the thread with 'approved'.
    4. Graph reaches terminal state and final state is verified.
    """
    # 1. Deterministically mock LLM evaluators to avoid flaky local GPU latency
    mock_injection = MagicMock()
    mock_injection.invoke.return_value = InjectionCheck(is_safe=True)

    mock_doc_grader = MagicMock()
    mock_doc_grader.invoke.return_value = GradeDocuments(binary_score="yes")

    mock_hallucination = MagicMock()
    mock_hallucination.invoke.return_value = GradeHallucinations(binary_score="yes")

    mock_relevance = MagicMock()
    mock_relevance.invoke.return_value = GradeAnswerRelevance(binary_score="yes")

    def structured_output_side_effect(schema):
        if schema == InjectionCheck:
            return mock_injection
        elif schema == GradeDocuments:
            return mock_doc_grader
        elif schema == GradeHallucinations:
            return mock_hallucination
        elif schema == GradeAnswerRelevance:
            return mock_relevance
        mock_generic = MagicMock()
        mock_generic.invoke.return_value = MagicMock(intent="rag")
        return mock_generic

    mocker.patch(
        "langchain_ollama.ChatOllama.with_structured_output",
        side_effect=structured_output_side_effect
    )

    # 2. Setup PostgreSQL Checkpointer
    checkpointer = PostgresSaver(postgres_pool)
    checkpointer.setup()
    app = workflow.compile(checkpointer=checkpointer)

    unique_thread_id = f"test-e2e-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": unique_thread_id}}

    inputs = {
        "question": "What is the rollback SLA? Contact devops@internal.io",
        "original_question": "",
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
        "history": [],
        "telemetry": {}
    }

    # 3. Step 1: Run graph until the Human-in-the-Loop breakpoint
    SEMANTIC_CACHE.clear()
    for _ in app.stream(inputs, config=config):
        pass

    # 4. Verify thread is paused and persisted in PostgreSQL
    paused_state = app.get_state(config)
    assert paused_state.next == ("human_approval",), (
        f"Expected next node to be ('human_approval',), got {paused_state.next}"
    )
    assert len(paused_state.values.get("documents", [])) > 0
    assert paused_state.values.get("generation") != ""

    # 5. Step 2: Use approve.py discovery logic
    pending_threads = list_pending_threads(postgres_pool)
    assert unique_thread_id in pending_threads, (
        f"Thread {unique_thread_id} should be visible in pending list, found: {pending_threads}"
    )

    # 6. Step 3: Inspect thread payload through approve.py
    inspected_state = inspect_thread(app, unique_thread_id)
    assert inspected_state is not None
    assert inspected_state.values.get("original_question") != ""

    # 7. Step 4: Resume execution via approve.py operator function
    process_decision(app, unique_thread_id, decision="approved")

    # 8. Step 5: Verify finalized state in PostgreSQL
    final_state = app.get_state(config)
    # Graph has reached END -> next nodes tuple is empty
    assert final_state.next == ()
    assert any("Assistant:" in entry for entry in final_state.values.get("history", []))

    # Verify semantic cache was updated upon approval
    import hashlib
    sanitized_q = final_state.values.get("sanitized_question")
    query_key = hashlib.sha256(sanitized_q.strip().lower().encode()).hexdigest()
    assert query_key in SEMANTIC_CACHE
