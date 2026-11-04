from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from rag_lab.pipeline.graph import workflow
from rag_lab.pipeline.state import GraphState
from rag_lab.storage.checkpointer import get_checkpointer, fetch_checkpoint_threads
from conftest import token_for

# The requester (carol) and the reviewer (bob) are different accounts, as separation of duties requires.
REVIEW = {"decision": "approved", "approver_token": token_for("bob"), "reason": "automated check"}


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


def test_full_pipeline_to_postgres_and_approve_cli(postgres_uri, rollback_index):
    """The production PostgreSQL path: encrypted, thread-bound saver from open_checkpointer, shared pool reuse."""
    import uuid

    from rag_lab.storage.checkpointer import close_checkpointer, open_checkpointer, shared_pool

    thread = f"test-integration-{uuid.uuid4().hex}"
    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    saver = open_checkpointer(postgres_uri)
    try:
        engine = workflow.compile(checkpointer=saver)
        for _ in engine.stream(_inputs(thread), config=config):
            pass
        assert engine.get_state(config).next == ("human_approval",)

        pool = shared_pool(saver)
        assert pool is not None
        assert thread in fetch_checkpoint_threads(uri=postgres_uri, prefix="test-integration-", pool=pool)
        assert thread in fetch_checkpoint_threads(uri=postgres_uri, prefix="test-integration-")

        for _ in engine.stream(Command(resume=REVIEW), config=config):
            pass
        final_state = engine.get_state(config)
        assert final_state.next == ()
        assert "rollback sla is strictly under 5 minutes" in final_state.values.get("generation", "").lower()
    finally:
        close_checkpointer(saver)
