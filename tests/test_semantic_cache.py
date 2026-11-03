"""Tests for persistent, clearance-partitioned semantic and vector-backed cache."""
from contextlib import closing
import sqlite3
from pathlib import Path
import pytest

from semantic_cache import (
    _L1_CITATIONS,
    _L1_GENERATIONS,
    _cosine_similarity,
    _pack_vector,
    _unpack_vector,
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
    assert ckey, "store_cache returns the key it stored under"

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


def test_hierarchical_clearance_exact_match(tmp_path: Path):
    db_file = str(tmp_path / "cache.db")

    # Store a public answer
    store_cache("ops", "What is the deploy SLA?", "public", "10 minutes", db_path=db_file)

    # Carol with internal clearance should get an instant exact match on the public answer
    hit = lookup_cache("ops", "What is the deploy SLA?", clearance="internal", db_path=db_file)
    assert hit["hit"] is True
    assert hit["hit_type"] == "exact"
    assert hit["generation"] == "10 minutes"


def test_prune_stale_cache_and_retention(tmp_path: Path, monkeypatch):
    import semantic_cache
    from datetime import datetime, UTC, timedelta

    db_file = str(tmp_path / "cache.db")

    # Mock index version to "v1"
    monkeypatch.setattr(semantic_cache, "_index_version", lambda ws: "v1")
    store_cache("ops", "question 1", "public", "answer 1", db_path=db_file)

    # Re-indexing advances version to "v2"
    monkeypatch.setattr(semantic_cache, "_index_version", lambda ws: "v2")
    removed_stale = semantic_cache.prune_stale_cache("ops", db_path=db_file)
    assert removed_stale == 1
    assert lookup_cache("ops", "question 1", "public", db_path=db_file)["hit"] is False

    # Retention pruning
    store_cache("ops", "question 2", "public", "answer 2", db_path=db_file)
    future = datetime.now(UTC) + timedelta(days=200)
    removed_ret = semantic_cache.prune_cache(retention_days=183, db_path=db_file, now=future)
    assert removed_ret == 1


def test_cache_stats_and_malformed_vector_handling(tmp_path: Path):
    from semantic_cache import _unpack_vector, cache_stats

    assert _unpack_vector(b"") == []
    assert _unpack_vector(b"123") == []  # Not a multiple of 4 bytes

    db_file = str(tmp_path / "cache.db")
    store_cache("ops", "q1", "public", "a1", db_path=db_file)
    store_cache("finance", "q2", "public", "a2", db_path=db_file)

    st = cache_stats(db_path=db_file)
    assert st["total_entries"] == 2
    assert "ops" in st["workspaces"]
    assert "finance" in st["workspaces"]
    assert st["bytes"] > 0


def test_cache_encryption_at_rest_and_row_aad_binding(tmp_path: Path):
    db_file = str(tmp_path / "cache.db")
    secret_answer = "Project-Omega-Confidential-998811"

    store_cache(
        workspace_id="ops",
        query="What is the Omega code?",
        clearance="secret",
        generation=secret_answer,
        citations=["omega.md"],
        db_path=db_file,
    )

    with closing(sqlite3.connect(db_file)) as conn, conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw_bytes = Path(db_file).read_bytes()
    assert secret_answer.encode("utf-8") not in raw_bytes
    assert b"What is the Omega code?" not in raw_bytes

    # Tamper with row clearance in SQLite directly (attempting to downgrade secret row to public):
    # AAD includes clearance, so decryption fails and the tampered row is refused!
    _L1_GENERATIONS.clear()
    _L1_CITATIONS.clear()
    with closing(sqlite3.connect(db_file)) as conn, conn:
        conn.execute("UPDATE semantic_cache SET clearance = 'public'")
        conn.commit()

    tampered_lookup = lookup_cache(
        "ops",
        "What is the Omega code?",
        clearance="public",
        allowed_clearances=["public"],
        db_path=db_file,
    )
    assert tampered_lookup["hit"] is False


def test_semantic_cache_numeric_and_negation_signature_guard(tmp_path: Path):
    """Near-identical questions with different numbers or negations must never collide on semantic lookup."""
    db_file = str(tmp_path / "cache.db")

    store_cache(
        workspace_id="default",
        query="What is the rollback SLA for tier 1 services?",
        clearance="public",
        generation="Tier 1 rollback SLA is 5 minutes.",
        citations=["sla.md"],
        db_path=db_file,
        compute_vector=True,
    )

    # Different number ("tier 2" vs "tier 1"): must miss even at low cosine threshold
    num_mismatch = lookup_cache(
        "default",
        "What is the rollback SLA for tier 2 services?",
        clearance="public",
        threshold=0.70,
        db_path=db_file,
    )
    assert num_mismatch["hit"] is False

    # Negation mismatch ("not" vs affirmative): must miss
    neg_mismatch = lookup_cache(
        "default",
        "What is not the rollback SLA for tier 1 services?",
        clearance="public",
        threshold=0.70,
        db_path=db_file,
    )
    assert neg_mismatch["hit"] is False


def test_unreviewed_cache_entries_withheld_from_users_requiring_human_review(tmp_path: Path):
    db_file = str(tmp_path / "cache.db")

    store_cache(
        workspace_id="default",
        query="What is the deploy window?",
        clearance="public",
        generation="Tuesday 02:00 UTC.",
        citations=["deploy.md"],
        reviewed=False,
        approved_by="auto_approve:alice",
        db_path=db_file,
    )

    # User without auto_approve (include_unreviewed=False) misses unreviewed entry
    strict_res = lookup_cache(
        "default",
        "What is the deploy window?",
        clearance="public",
        include_unreviewed=False,
        db_path=db_file,
    )
    assert strict_res["hit"] is False

    # User with auto_approve (include_unreviewed=True) hits it
    auto_res = lookup_cache(
        "default",
        "What is the deploy window?",
        clearance="public",
        include_unreviewed=True,
        db_path=db_file,
    )
    assert auto_res["hit"] is True

