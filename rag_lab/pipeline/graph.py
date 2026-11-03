"""The Self-RAG state machine: routing decisions and the graph wiring.

Stages live in entry, retrieval, generation, grading, and review; docs/architecture.mmd is generated
from this module by `make graph`.
"""
from __future__ import annotations

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from rag_lab.pipeline.context import contextualize_node
from rag_lab.pipeline.entry import cache_lookup_node, chitchat_node, guardrail_input_node, router_node
from rag_lab.pipeline.generation import generate_node
from rag_lab.pipeline.grading import grade_answer_relevance_node, grade_hallucination_node
from rag_lab.pipeline.retrieval import grade_documents_node, hybrid_retrieval_node, query_rewrite_node, sanitize_context_node
from rag_lab.pipeline.review import human_in_the_loop_node
from rag_lab.pipeline.state import GraphState, StateInput


def route_safety(state: StateInput) -> str:
    return "pass" if state.get("is_safe", True) else "fail"


def route_cache(state: StateInput) -> str:
    return "hit" if state.get("cache_hit", False) else "miss"


def route_intent(state: StateInput) -> str:
    return state.get("intent", "rag")


def route_doc_grade(state: StateInput) -> str:
    status = state.get("grade_status", "approved")
    if status == "circuit_break":
        return "circuit_break"
    if status == "rewrite":
        return "rewrite"
    return "approved"


def route_hallucination(state: StateInput) -> str:
    status = state.get("hallucination_status", "grounded")
    if status == "circuit_break":
        return "circuit_break"
    if status == "retry":
        return "retry"
    return "grounded"


def route_relevance(state: StateInput) -> str:
    status = state.get("relevance_status", "relevant")
    if status == "circuit_break":
        return "circuit_break"
    if status == "irrelevant":
        return "irrelevant"
    return "relevant"


builder = StateGraph(GraphState)


builder.add_node("guardrail", guardrail_input_node)


builder.add_node("contextualize", contextualize_node)
builder.add_edge("contextualize", "cache_lookup")
builder.add_node("cache_lookup", cache_lookup_node)


builder.add_node("router", router_node)


builder.add_node("chitchat", chitchat_node)


builder.add_node("retrieve", hybrid_retrieval_node)


builder.add_node("sanitize_context", sanitize_context_node)


builder.add_node("grade_docs", grade_documents_node)


builder.add_node("rewrite", query_rewrite_node)


builder.add_node("generate", generate_node)


builder.add_node("grade_hallucination", grade_hallucination_node)


builder.add_node("grade_relevance", grade_answer_relevance_node)


builder.add_node("human_approval", human_in_the_loop_node)


builder.add_edge(START, "guardrail")


builder.add_conditional_edges(
    "guardrail",
    route_safety,
    {"pass": "contextualize", "fail": END}
)


builder.add_conditional_edges(
    "cache_lookup",
    route_cache,
    {"hit": END, "miss": "router"}
)


builder.add_conditional_edges(
    "router",
    route_intent,
    {"chitchat": "chitchat", "rag": "retrieve"}
)


builder.add_edge("chitchat", END)


builder.add_edge("retrieve", "sanitize_context")


builder.add_edge("sanitize_context", "grade_docs")


builder.add_conditional_edges(
    "grade_docs",
    route_doc_grade,
    {"approved": "generate", "rewrite": "rewrite", "circuit_break": "generate"}
)


builder.add_edge("rewrite", "retrieve")


builder.add_edge("generate", "grade_hallucination")


builder.add_conditional_edges(
    "grade_hallucination",
    route_hallucination,
    {"grounded": "grade_relevance", "retry": "generate", "circuit_break": "human_approval"}
)


builder.add_conditional_edges(
    "grade_relevance",
    route_relevance,
    {"relevant": "human_approval", "irrelevant": "rewrite", "circuit_break": "human_approval"}
)


builder.add_edge("human_approval", END)


workflow = builder
