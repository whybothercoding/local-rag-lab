"""Tests for persistent, clearance-partitioned semantic and vector-backed cache."""
import os
import sqlite3
from pathlib import Path
import pytest

from access import add_user, issue_token
from app import cache_lookup_node, human_in_the_loop_node, workflow
from conftest import token_for
from semantic_cache import (
    _L1_CITATIONS,
    _L1_GENERATIONS,
    _cosine_similarity,
    _pack_vector,
    _unpack_vector,
    cache_key,
    clear_cache,
    clear_workspace_cache,
    lookup_cache,
    store_cache,
    PersistentSemanticCache,
    PersistentCitationsCache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    clear_cache()
    yield
    clear_cache()


def test_vector_packing_and_cosine_similarity():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]

    packed = _pack_vector(vec_a)
    unpacked = _unpack_vector(packed)
    assert len(unpacked) == 3
    assert abs(unpacked[0] - 1.0) < 1e-6

    assert abs(_cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6
    assert abs(_cosine_similarity(vec_a, vec_c) - 0.0) < 1e-6
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cache_persists_across_l1_in_memory_eviction(tmp_path: Path):
    db_file = str(tmp_path / "cache.db")

    ckey = store_cache(
        workspace_id="default",
        query="What is the rollback SLA?",
        clearance="public",
        generation="Under 5 minutes.",
        citations=["policy.md (chunk 0)"],
        db_path=db_file,
    )

    # 1. Hit from L1 memory
    hit_l1 = lookup_cache("default", "What is the rollback SLA?", "public", db_path=db_file)
    assert hit_l1["hit"] is True
    assert hit_l1["hit_type"] == "exact"
    assert hit_l1["generation"] == "Under 5 minutes."
    assert hit_l1["citations"] == ["policy.md (chunk 0)"]

    # 2. Simulate process restart / L1 memory wipe
    _L1_GENERATIONS.clear()
    _L1_CITATIONS.clear()

    # Must still hit from SQLite persistence
    hit_l2 = lookup_cache("default", "What is the rollback SLA?", "public", db_path=db_file)
    assert hit_l2["hit"] is True
    assert hit_l2["generation"] == "Under 5 minutes."
    assert hit_l2["citations"] == ["policy.md (chunk 0)"]


def test_semantic_vector_cache_hit_on_paraphrased_query(tmp_path: Path):
    """Verifies that asking a semantically equivalent question trips a vector cache hit."""
    db_file = str(tmp_path / "cache.db")

    # Store answer with real fastembed embedding vector
    store_cache(
        workspace_id="default",
        query="What is the rollback SLA limit for production?",
        clearance="public",
        generation="Production rollback SLA is strictly under 5 minutes.",
        citations=["runbook.md"],
        db_path=db_file,
        compute_vector=True,
    )

    # Query with exact wording: exact hit
    exact_res = lookup_cache("default", "What is the rollback SLA limit for production?", "public", db_path=db_file)
    assert exact_res["hit"] is True
    assert exact_res["hit_type"] == "exact"

    # Query with paraphrased wording: trips semantic hit
    paraphrase = "What is our production rollback SLA timeframe?"
    semantic_res = lookup_cache(
        "default",
        paraphrase,
        "public",
        threshold=0.85,
        db_path=db_file,
    )
    assert semantic_res["hit"] is True
    assert semantic_res["hit_type"] == "semantic"
    assert semantic_res["similarity"] >= 0.85
    assert semantic_res["generation"] == "Production rollback SLA is strictly under 5 minutes."
    assert semantic_res["citations"] == ["runbook.md"]


def test_semantic_cache_zero_trust_clearance_isolation(tmp_path: Path):
    """A secret answer must never be returned to a public user even with an exact or semantic query match."""
    db_file = str(tmp_path / "cache.db")

    store_cache(
        workspace_id="default",
        query="What is the vault master passphrase?",
        clearance="secret",
        generation="Vault passphrase is TopSecret-XYZ-999.",
        citations=["keys.md"],
        db_path=db_file,
        compute_vector=True,
    )

    # Public user exact match attempt
    pub_exact = lookup_cache(
        "default",
        "What is the vault master passphrase?",
        clearance="public",
        allowed_clearances=["public"],
        db_path=db_file,
    )
    assert pub_exact["hit"] is False

    # Public user semantic paraphrase attempt
    pub_semantic = lookup_cache(
        "default",
        "Tell me the vault passphrase",
        clearance="public",
        allowed_clearances=["public"],
        threshold=0.70,
        db_path=db_file,
    )
    assert pub_semantic["hit"] is False

    # Secret user match succeeds
    sec_hit = lookup_cache(
        "default",
        "What is the vault master passphrase?",
        clearance="secret",
        allowed_clearances=["public", "internal", "confidential", "secret"],
        db_path=db_file,
    )
    assert sec_hit["hit"] is True
    assert "TopSecret-XYZ-999" in sec_hit["generation"]


def test_clear_workspace_cache_evicts_only_target_workspace(tmp_path: Path):
    db_file = str(tmp_path / "cache.db")

    store_cache("finance", "quarterly profit", "public", "10 million", db_path=db_file)
    store_cache("engineering", "deploy schedule", "public", "Tuesday 10am", db_path=db_file)

    assert lookup_cache("finance", "quarterly profit", "public", db_path=db_file)["hit"] is True
    assert lookup_cache("engineering", "deploy schedule", "public", db_path=db_file)["hit"] is True

    clear_workspace_cache("finance", db_path=db_file)

    assert lookup_cache("finance", "quarterly profit", "public", db_path=db_file)["hit"] is False
    assert lookup_cache("engineering", "deploy schedule", "public", db_path=db_file)["hit"] is True


def test_dict_proxy_interfaces():
    cache = PersistentSemanticCache()
    cites = PersistentCitationsCache()

    cache["k1"] = "gen1"
    cites["k1"] = ["cite1"]

    assert "k1" in cache
    assert cache["k1"] == "gen1"
    assert cache.get("k1") == "gen1"
    assert cites["k1"] == ["cite1"]

    assert len(cache) >= 1
    assert "k1" in list(cache)

    del cache["k1"]
    assert "k1" not in cache
