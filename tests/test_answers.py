"""Answer-correctness controls: abstention, cache invalidation, labelled degradation, provider, and feedback."""
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from rag_lab.pipeline.state import ABSTAIN_MESSAGE
from rag_lab.pipeline.graph import workflow
from rag_lab.retrieval.semantic_cache import cache_key
from rag_lab.pipeline.entry import cache_lookup_node, SEMANTIC_CACHE
from rag_lab.pipeline.generation import generate_node
from rag_lab.pipeline.retrieval import hybrid_retrieval_node
from rag_lab.security.audit import list_rejections, record
from conftest import token_for
from rag_lab.ingestion.ingest import ingest_workspace
from rag_lab.pipeline.llm_provider import OpenAICompatibleChat
from rag_lab.config import settings

ROLLBACK_QUESTION = "What is the rollback SLA?"
OFF_TOPIC_QUESTION = "What is the office dress code policy?"


def _retrieval_state(question: str, workspace: str = "default") -> dict:
    return {"sanitized_question": question, "workspace_id": workspace, "telemetry": {}, "degraded": []}


def _run_until_review(engine, thread: str, question: str) -> dict:
    config = {"configurable": {"thread_id": thread}}
    inputs = {
        "workspace_id": "default",
        "session_id": thread,
        "principal_token": token_for("carol"),
        "question": question,
        "original_question": question,
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
    for _ in engine.stream(inputs, config=config):
        pass
    state = engine.get_state(config)
    assert state.next == ("human_approval",)
    return state.values


# --- G16: abstention --------------------------------------------------------

def test_off_topic_question_abstains_and_still_goes_to_review(rollback_index):
    values = _run_until_review(workflow.compile(checkpointer=MemorySaver()), "t-off-topic", OFF_TOPIC_QUESTION)
    assert values["generation"] == ABSTAIN_MESSAGE
    assert values["telemetry"]["abstained"] is True
    assert values["telemetry"]["relevant"] is False


def test_relevance_gate_withholds_documents_for_off_topic_questions(rollback_index):
    state = hybrid_retrieval_node(_retrieval_state(OFF_TOPIC_QUESTION))
    assert state["documents"] == []
    assert state["telemetry"]["relevant"] is False


def test_relevant_question_is_answered_from_its_citation(rollback_index):
    state = hybrid_retrieval_node(_retrieval_state(ROLLBACK_QUESTION))
    assert state["telemetry"]["relevant"] is True
    assert state["citations"] and state["citations"][0].startswith("[")
    assert "5 minutes" in state["documents"][0]


# --- G17: cache invalidation ------------------------------------------------

def test_reindexing_changes_the_cache_key(rollback_index):
    before = cache_key("default", ROLLBACK_QUESTION)
    (rollback_index / "company_policy.md").write_text("Rollback SLA is strictly under 3 minutes.", encoding="utf-8")
    assert ingest_workspace("default", str(rollback_index)).ok
    assert cache_key("default", ROLLBACK_QUESTION) != before


def test_cached_answer_is_not_served_after_reindexing(rollback_index):
    SEMANTIC_CACHE.clear()
    state = {"sanitized_question": ROLLBACK_QUESTION, "workspace_id": "default", "telemetry": {}}
    SEMANTIC_CACHE[cache_key("default", ROLLBACK_QUESTION)] = "Old answer from the previous index."
    assert cache_lookup_node(state)["cache_hit"] is True

    (rollback_index / "company_policy.md").write_text("Rollback SLA is strictly under 3 minutes.", encoding="utf-8")
    assert ingest_workspace("default", str(rollback_index)).ok
    assert cache_lookup_node(state)["cache_hit"] is False


# --- G18: failures are recorded, never hidden -------------------------------

def test_dense_outage_is_recorded_and_sparse_results_still_return(rollback_index, monkeypatch):
    def broken_client(*args, **kwargs):
        raise RuntimeError("vector store offline")

    monkeypatch.setattr("chromadb.PersistentClient", broken_client)  # what ChromaStore opens
    state = hybrid_retrieval_node(_retrieval_state(ROLLBACK_QUESTION))
    assert "dense_retrieval_unavailable: RuntimeError" in state["degraded"]
    assert state["documents"], "keyword retrieval should still answer from the index"


def test_missing_index_is_reported_as_not_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "empty-chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "empty-bm25"))
    state = hybrid_retrieval_node(_retrieval_state(ROLLBACK_QUESTION, workspace="ghost"))
    assert any(note.startswith("not_indexed") for note in state["degraded"])
    assert state["documents"] == []


def test_model_failure_gives_a_labelled_extractive_answer(monkeypatch):
    class BrokenModel:
        def invoke(self, prompt):
            raise TimeoutError("model did not respond")

    monkeypatch.setattr("rag_lab.pipeline.llm_provider.get_chat_llm", lambda: BrokenModel())
    state = generate_node({
        "documents": ["Rollback SLA is strictly under 5 minutes."],
        "sanitized_question": "What is the rollback SLA?",
        "degraded": [],
        "telemetry": {},
    })
    assert "5 minutes" in state["generation"]
    assert state["telemetry"]["generation_fallback"] is True
    assert any(note.startswith("generation_fallback: TimeoutError") for note in state["degraded"])


def test_seeded_documents_cannot_bypass_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "empty-chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "empty-bm25"))
    state = hybrid_retrieval_node({
        "sanitized_question": ROLLBACK_QUESTION,
        "workspace_id": "ghost",
        "documents": ["INJECTED TEXT"],
        "citations": ["[forged] policy.md"],
        "telemetry": {},
        "degraded": [],
    })
    assert "INJECTED TEXT" not in state["documents"]
    assert state["documents"] == []


# --- G21: OpenAI-compatible local provider ----------------------------------

def test_openai_compatible_provider_speaks_chat_completions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Five minutes."}}]})

    llm = OpenAICompatibleChat(
        base_url="http://local-llm.test/v1",
        model_name="local-model",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    reply = llm.invoke([SystemMessage(content="Be brief."), HumanMessage(content="Rollback?")])

    assert reply.content == "Five minutes."
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
    assert seen["body"]["model"] == "local-model"


def test_openai_compatible_provider_surfaces_server_errors():
    llm = OpenAICompatibleChat(
        base_url="http://local-llm.test/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        llm.invoke([HumanMessage(content="Rollback?")])


# --- G22: rejection feedback ------------------------------------------------

def test_rejections_are_listed_with_their_reasons(tmp_path):
    path = str(tmp_path / "audit.db")
    record("hitl_decision", workspace_id="ops", session_id="s1",
           data={"decision": "rejected", "released": False, "approver": "bob", "reason": "Missing the sign-off clause"},
           path=path)
    record("hitl_decision", workspace_id="ops", session_id="s2",
           data={"decision": "approved", "released": True, "approver": "bob", "reason": ""}, path=path)

    rejections = list_rejections(path=path)
    assert len(rejections) == 1
    assert rejections[0]["data"]["reason"] == "Missing the sign-off clause"


def test_auto_approve_grant_releases_answer_without_human_pause(rollback_index):
    from rag_lab.security.access import set_grant

    set_grant("dave", "default", ["query", "auto_approve"])
    try:
        engine = workflow.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "auto-approve-thread-1"}}
        inputs = {
            "workspace_id": "default",
            "session_id": "auto-approve-thread-1",
            "principal_token": token_for("dave"),
            "question": ROLLBACK_QUESTION,
            "original_question": ROLLBACK_QUESTION,
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

        for _ in engine.stream(inputs, config=config):
            pass

        state = engine.get_state(config)
        assert state.next == ()
        assert "rollback sla is strictly under 5 minutes" in state.values.get("generation", "").lower()
    finally:
        set_grant("dave", "default", [])
