import pytest
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from app import workflow, DB_URI, GraphState
from checkpointer import get_checkpointer, fetch_checkpoint_threads
from conftest import token_for

# The requester (carol) and the reviewer (bob) are different accounts, as separation of duties requires.
REVIEW = {"decision": "approved", "approver_token": token_for("bob"), "reason": "automated check"}


def is_postgres_available() -> bool:
    """Quick check to avoid 30-second pytest hanging if PostgreSQL container is off."""
    try:
        with ConnectionPool(conninfo=DB_URI, min_size=1, max_size=1, timeout=1.5, kwargs={"autocommit": True}) as pool:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return True
    except Exception:
        return False


def _inputs(session: str) -> GraphState:
    return {
        "workspace_id": "default",
        "session_id": session,
        "principal_token": token_for("carol"),
        "question": "What is our company rollback SLA?",
        "original_question": "What is our company rollback SLA?",
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
        "telemetry": {},
        "degraded": [],
    }


def test_full_pipeline_to_sqlite_and_approve_cli(tmp_path, rollback_index):
    sqlite_uri = f"sqlite:///{tmp_path / 'test_checkpoints.db'}"
    config: RunnableConfig = {"configurable": {"thread_id": "test-sqlite-thread-001"}}

    with get_checkpointer(uri=sqlite_uri) as checkpointer:
        engine = workflow.compile(checkpointer=checkpointer)

        # Step until interrupt
        for _ in engine.stream(_inputs("test-sqlite-thread-001"), config=config):
            pass

        state = engine.get_state(config)
        assert state.next == ("human_approval",)
        assert state.values["citations"], "answer should be grounded in the indexed document"

        # Confirm thread is discoverable by checkpointer inspection
        threads = fetch_checkpoint_threads(uri=sqlite_uri)
        assert "test-sqlite-thread-001" in threads

        # Resume with a named reviewer
        for _ in engine.stream(Command(resume=REVIEW), config=config):
            pass

        final_state = engine.get_state(config)
        assert final_state.next == ()
        assert "rollback sla is strictly under 5 minutes" in final_state.values.get("generation", "").lower()


@pytest.mark.skipif(not is_postgres_available(), reason="PostgreSQL container not reachable on localhost:5432")
def test_full_pipeline_to_postgres_and_approve_cli(rollback_index):
    config: RunnableConfig = {"configurable": {"thread_id": "test-integration-thread-001"}}

    with ConnectionPool(conninfo=DB_URI, max_size=5, open=True, kwargs={"autocommit": True}) as pool:
        checkpointer = PostgresSaver(pool)  # type: ignore[arg-type]
        checkpointer.setup()
        engine = workflow.compile(checkpointer=checkpointer)

        # Step until interrupt
        for _ in engine.stream(_inputs("test-integration-thread-001"), config=config):
            pass

        state = engine.get_state(config)
        assert state.next == ("human_approval",)

        # Resume with a named reviewer
        for _ in engine.stream(Command(resume=REVIEW), config=config):
            pass

        final_state = engine.get_state(config)
        assert final_state.next == ()
        assert "rollback sla is strictly under 5 minutes" in final_state.values.get("generation", "").lower()
