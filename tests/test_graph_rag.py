"""Tests for the GraphRAG Knowledge Graph augmentation layer."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.ingestion.graph_builder import build_workspace_graph, extract_triplets_from_text
from rag_lab.pipeline.retrieval import hybrid_retrieval_node, reciprocal_rank_fusion
from rag_lab.retrieval.graph_store import KnowledgeGraph
from rag_lab.retrieval.sparse_search import index_sparse_chunks


def test_knowledge_graph_basic_operations() -> None:
    kg = KnowledgeGraph("test_ws")
    kg.add_entity("Eleni Vance", "PERSON", aliases=["E. Vance"])
    kg.add_entity("Infrastructure Security Lead", "ROLE")
    kg.add_relationship(
        "Eleni Vance",
        "Infrastructure Security Lead",
        "HOLDS_ROLE",
        source_chunk="chunk_1",
    )
    kg.add_relationship(
        "Infrastructure Security Lead",
        "Production Database Keys",
        "CAN_AUTHORIZE",
        source_chunk="chunk_2",
    )

    # Alias and case-insensitive matching
    matched = kg.find_entities_in_text("Does E. Vance have authorization?")
    assert len(matched) == 1
    assert matched[0].name == "Eleni Vance"

    # Multi-hop traversal (2 hops)
    traversal = kg.traverse([matched[0].id], max_hops=2)
    assert "infrastructure_security_lead" in traversal.visited_entity_ids
    assert "production_database_keys" in traversal.visited_entity_ids
    assert "chunk_1" in traversal.chunk_ids
    assert "chunk_2" in traversal.chunk_ids
    assert len(traversal.paths) >= 2


def test_knowledge_graph_bidirectional_traversal() -> None:
    kg = KnowledgeGraph("test_ws")
    kg.add_relationship("Alice", "SecOps Lead", "HOLDS_ROLE", source_chunk="c1")
    kg.add_relationship("SecOps Lead", "Production DB", "CAN_AUTHORIZE", source_chunk="c2")

    # Traverse starting from target (inbound reverse lookup)
    res = kg.traverse(["production_db"], max_hops=2)
    assert "secops_lead" in res.visited_entity_ids
    assert "alice" in res.visited_entity_ids
    assert "c1" in res.chunk_ids
    assert "c2" in res.chunk_ids


def test_knowledge_graph_serialization(tmp_path: Path) -> None:
    kg = KnowledgeGraph("export_ws")
    kg.add_relationship("Dr. Smith", "Director", "HOLDS_ROLE", source_chunk="doc_1")
    save_file = tmp_path / "graph.json"
    kg.save(save_file)

    loaded = KnowledgeGraph.load(save_file)
    assert loaded.workspace_id == "export_ws"
    assert "dr_smith" in loaded.entities
    assert len(loaded.relationships) == 1
    assert loaded.relationships[0].predicate == "HOLDS_ROLE"


def test_the_sparse_index_refuses_mismatched_lists(tmp_path: Path) -> None:
    """zip() would silently drop the tail chunks; a length mismatch must be refused instead."""
    with pytest.raises(ValueError, match="length"):
        index_sparse_chunks("ws", ["a", "b"], ["one.md"], ["first", "second"], base_dir=str(tmp_path))


def test_an_edge_seen_in_two_chunks_keeps_both_provenances() -> None:
    kg = KnowledgeGraph("ws")
    kg.add_relationship("Eleni Vance", "Infrastructure Security Lead", "HOLDS_ROLE", source_chunk="chunk_a")
    kg.add_relationship("Eleni Vance", "Infrastructure Security Lead", "HOLDS_ROLE", source_chunk="chunk_b")
    matched = kg.find_entities_in_text("Does Eleni Vance approve?")[0]
    traversal = kg.traverse([matched.id], max_hops=2)
    assert set(traversal.chunk_ids) == {"chunk_a", "chunk_b"}


def test_lowercase_sentence_fragments_do_not_become_junk_triplets() -> None:
    """The relation patterns are anchored on a capital: a lowercase mid-sentence fragment
    ("the deadline is Friday") must not yield a PERSON->ROLE triplet."""
    assert extract_triplets_from_text("Remember: the deadline is Friday, not Thursday.", "c") == []
    triples = extract_triplets_from_text("Eleni Vance holds the title of Infrastructure Security Lead.", "c")
    assert any(tr["predicate"] == "HOLDS_ROLE" and "Eleni Vance" in tr["source"] for tr in triples)


def test_multilingual_graph_pattern_extraction() -> None:
    # English
    t1 = "Eleni Vance holds the title of Infrastructure Security Lead."
    triples1 = extract_triplets_from_text(t1, "c_en")
    assert any(tr["predicate"] == "HOLDS_ROLE" and "Eleni Vance" in tr["source"] for tr in triples1)

    # German
    t2 = "Dr. Schmidt ist der Datenschutzbeauftragte."
    triples2 = extract_triplets_from_text(t2, "c_de")
    assert any("Dr. Schmidt" in tr["source"] for tr in triples2)

    # French
    t3 = "Claire Dupont est la Directrice de la Sécurité."
    triples3 = extract_triplets_from_text(t3, "c_fr")
    assert any("Claire Dupont" in tr["source"] for tr in triples3)

    # Greek
    t4 = "Ο Γιάννης Παπαδόπουλος είναι ο Επικεφαλής Ασφάλειας."
    triples4 = extract_triplets_from_text(t4, "c_el")
    assert any("Γιάννης Παπαδόπουλος" in tr["source"] for tr in triples4)


def test_reciprocal_rank_fusion_with_graph_results() -> None:
    dense = [{"id": "d1", "content": "Dense 1", "citation": "[d1]"}]
    sparse = [{"id": "s1", "content": "Sparse 1", "citation": "[s1]"}]
    graph = [{"id": "g1", "content": "Graph 1", "citation": "[g1]"}]

    fused = reciprocal_rank_fusion(dense, sparse, graph_results=graph, top_n=3)
    fused_ids = [item["id"] for item in fused]
    assert "g1" in fused_ids
    assert len(fused) == 3


def test_end_to_end_graph_multi_hop_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws_id = "multihop_test"
    bm25_dir = tmp_path / "bm25"
    graphs_dir = tmp_path / "graphs"
    bm25_dir.mkdir()
    graphs_dir.mkdir()

    monkeypatch.setattr("rag_lab.config.settings.bm25_dir", str(bm25_dir))
    monkeypatch.setattr("rag_lab.config.settings.graphs_dir", str(graphs_dir))
    monkeypatch.setattr("rag_lab.config.settings.enable_graph_rag", True)
    monkeypatch.setattr("rag_lab.config.settings.graph_max_hops", 2)
    monkeypatch.setattr("rag_lab.config.settings.enable_reranker", False)
    monkeypatch.setattr("rag_lab.config.settings.relevance_gate", "similarity")
    monkeypatch.setattr("rag_lab.config.settings.relevance_min_similarity", 0.1)

    # 3 Chunks establishing a 2-hop corporate chain:
    # Chunk A: Eleni Vance holds the position of Infrastructure Security Lead.
    # Chunk B: Production database disaster recovery keys may only be authorized by the Infrastructure Security Lead.
    # Chunk C: Disaster recovery request #4092 applies to Production database disaster recovery keys.
    chunks_data = [
        {
            "id": "chunk_a",
            "text": "Eleni Vance holds the title of Infrastructure Security Lead.",
            "source": "personnel.md",
        },
        {
            "id": "chunk_b",
            "text": "Production database disaster recovery keys may only be authorized by the Infrastructure Security Lead.",
            "source": "security_policy.md",
        },
        {
            "id": "chunk_c",
            "text": "Disaster recovery request #4092 applies to Production database disaster recovery keys.",
            "source": "tickets.md",
        },
    ]

    # Index chunks in SQLite FTS5
    index_sparse_chunks(
        ws_id,
        [c["id"] for c in chunks_data],
        [c["source"] for c in chunks_data],
        [c["text"] for c in chunks_data],
        base_dir=str(bm25_dir),
    )

    # Build and save knowledge graph
    kg = build_workspace_graph(ws_id, chunks_data)
    kg.save(graphs_dir / f"{ws_id}.json")

    # Run hybrid retrieval with GraphRAG enabled
    state = {
        "workspace_id": ws_id,
        "sanitized_question": "Can Eleni Vance authorize disaster recovery request #4092?",
        "principal": "auditor",
        "retrieval_strategy": "graph",
    }
    result = hybrid_retrieval_node(state)

    assert result["telemetry"]["relevant"] is True
    assert "graph_traversal" in result["telemetry"]
    graph_info = result["telemetry"]["graph_traversal"]
    assert len(graph_info["paths"]) > 0

    # Ensure all multi-hop bridging chunks are retrieved into context
    doc_text = " ".join(result["documents"])
    assert "Eleni Vance" in doc_text
    assert "Infrastructure Security Lead" in doc_text
    assert "recovery keys" in doc_text


def test_graphrag_vs_standard_rag_bridging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws_id = "comparison_ws"
    bm25_dir = tmp_path / "bm25"
    graphs_dir = tmp_path / "graphs"
    bm25_dir.mkdir()
    graphs_dir.mkdir()

    monkeypatch.setattr("rag_lab.config.settings.bm25_dir", str(bm25_dir))
    monkeypatch.setattr("rag_lab.config.settings.graphs_dir", str(graphs_dir))
    monkeypatch.setattr("rag_lab.config.settings.graph_max_hops", 2)
    monkeypatch.setattr("rag_lab.config.settings.enable_reranker", False)
    monkeypatch.setattr("rag_lab.config.settings.relevance_gate", "similarity")
    monkeypatch.setattr("rag_lab.config.settings.relevance_min_similarity", 0.0)
    monkeypatch.setattr("rag_lab.config.settings.retrieval_sparse_k", 2)
    monkeypatch.setattr("rag_lab.config.settings.retrieval_fusion_top_n", 2)

    chunks_data = [
        {
            "id": "chunk_a",
            "text": "Eleni Vance holds the title of Infrastructure Security Lead.",
            "source": "hr.md",
        },
        {
            "id": "chunk_b",
            "text": "Production database root keys may only be authorized by the Infrastructure Security Lead.",
            "source": "governance.md",
        },
        {
            "id": "chunk_c",
            "text": "Emergency access ticket #9901 references Production database root keys.",
            "source": "incidents.md",
        },
    ]

    index_sparse_chunks(
        ws_id,
        [c["id"] for c in chunks_data],
        [c["source"] for c in chunks_data],
        [c["text"] for c in chunks_data],
        base_dir=str(bm25_dir),
    )
    kg = build_workspace_graph(ws_id, chunks_data)
    kg.save(graphs_dir / f"{ws_id}.json")

    # 1. Standard Hybrid RAG without Graph:
    # Query mentions Eleni Vance and ticket #9901.
    # The bridging chunk (chunk_b) does NOT mention "Eleni Vance" or "ticket #9901",
    # so BM25 ranks chunk_a and chunk_c, omitting chunk_b from the top 2 candidates.
    monkeypatch.setattr("rag_lab.config.settings.enable_graph_rag", False)
    res_standard = hybrid_retrieval_node({
        "workspace_id": ws_id,
        "sanitized_question": "Does Eleni Vance have clearance to approve emergency access ticket #9901?",
        "principal": "auditor",
        "retrieval_strategy": "hybrid",
    })
    assert "governance.md" not in " ".join(res_standard["citations"])

    # 2. Graph-Augmented RAG (GraphRAG):
    # Traverses Eleni Vance -> Infrastructure Security Lead -> Production database root keys -> Emergency access ticket #9901
    # Resolves and retrieves the missing bridging chunk_b from governance.md!
    monkeypatch.setattr("rag_lab.config.settings.enable_graph_rag", True)
    res_graph = hybrid_retrieval_node({
        "workspace_id": ws_id,
        "sanitized_question": "Does Eleni Vance have clearance to approve emergency access ticket #9901?",
        "principal": "auditor",
        "retrieval_strategy": "graph",
    })
    assert "governance.md" in " ".join(res_graph["citations"])
    assert "Infrastructure Security Lead" in " ".join(res_graph["documents"])
