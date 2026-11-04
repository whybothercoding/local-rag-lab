"""The semantic cache's vector index: same answers as the old linear scan, never across clearances, never stale."""
import random
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import rag_lab.retrieval.semantic_cache as sc
from rag_lab.retrieval import cache_index
from rag_lab.retrieval.cache_index import PartitionIndex, unit
from rag_lab.security.access import allowed_clearances

DIM = 32
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]


class FakeEmbedder:
    """Deterministic vectors from text, so tests control exactly how similar two questions are."""

    def __init__(self):
        self.vectors: dict[str, list[float]] = {}

    def vector(self, text: str) -> list[float]:
        if text not in self.vectors:
            rng = np.random.default_rng(abs(hash(("seed", text))) % (2**32))
            v = rng.normal(size=DIM)
            self.vectors[text] = (v / np.linalg.norm(v)).tolist()
        return self.vectors[text]

    def near(self, text: str, base: str, noise: float, seed: int) -> None:
        v = np.array(self.vector(base)) + np.random.default_rng(seed).normal(size=DIM) * noise
        self.vectors[text] = (v / np.linalg.norm(v)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


@pytest.fixture
def fake(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr("rag_lab.retrieval.embedding_provider.get_embedding_model", lambda *a, **k: embedder)
    cache_index.forget()
    yield embedder
    cache_index.forget()


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cache.db")


def question(word: str, other: str = "") -> str:
    return f"What is the rule about {word} {other}?".replace("  ", " ")


def oracle(db_path, ws, clearance, q, q_vec, threshold, include_unreviewed):
    """The algorithm this index replaced: fetch, open and compare every candidate in Python."""
    authorized = allowed_clearances(clearance)
    version, cutoff = sc._index_version(ws), sc._retention_cutoff()
    marks = ", ".join("?" for _ in authorized)
    with sc._open_db(db_path) as conn:
        rows = conn.execute(
            f"SELECT {sc._COLUMNS} FROM semantic_cache WHERE workspace_id = ? AND index_version = ? "
            f"AND clearance IN ({marks}) AND query_vector IS NOT NULL AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT 2000;", [ws, version, *authorized, cutoff]).fetchall()
    entries = [e for e in (sc._open_row(r) for r in rows) if e and e["vector"]]
    if not include_unreviewed:
        entries = [e for e in entries if e["reviewed"]]
    signature = sc._signature(q)
    best, best_sim = None, -1.0
    for entry in entries:
        if sc._signature(entry["query"]) != signature:
            continue
        sim = sc._cosine_similarity(q_vec, entry["vector"])
        if sim > best_sim:
            best, best_sim = entry, sim
    return (best["key"], round(best_sim, 4)) if best is not None and best_sim >= threshold else None


# --- the same answers as the scan it replaces ----------------------------------------------------------------

def test_matches_the_linear_scan_on_a_randomised_store(db, fake):
    rng = random.Random(11)
    stored = []
    for i in range(90):
        q = question(rng.choice(WORDS), rng.choice(WORDS + ["", "not", "2024", "2025"]))
        if q in stored:
            continue
        stored.append(q)
        sc.store_cache(rng.choice(["ops", "ops", "hr"]), q, rng.choice(["public", "internal", "confidential", "secret"]),
                       f"answer to {q}", ["[c] d.md"], db_path=db, reviewed=rng.random() < 0.7)
    probes = []
    for n in range(40):
        base = rng.choice(stored)
        probe = f"What is the rule about {rng.choice(WORDS)} {rng.choice(WORDS)} variant {chr(97 + n % 26)}{chr(97 + n // 26)}?"
        fake.near(probe, base, rng.choice([0.05, 0.2, 0.6]), seed=n)
        probes.append(probe)
    checked = hits = 0
    for probe in probes:
        for ws in ("ops", "hr"):
            for clearance in ("public", "confidential", "secret"):
                for include_unreviewed in (True, False):
                    threshold = rng.choice([0.3, 0.6, 0.85])
                    expected = oracle(db, ws, clearance, probe, fake.embed_query(probe), threshold, include_unreviewed)
                    sc._l1_clear()
                    got = sc.lookup_cache(ws, probe, clearance, threshold=threshold, db_path=db,
                                          include_unreviewed=include_unreviewed)
                    checked += 1
                    if expected is None:
                        assert got["hit"] is False or got["hit_type"] == "exact", (probe, ws, clearance, threshold)
                    else:
                        hits += 1
                        assert got["hit"] and got["hit_type"] in ("semantic", "exact")
                        if got["hit_type"] == "semantic":
                            assert got["key"] == expected[0] and abs(got["similarity"] - expected[1]) < 1e-9
    assert checked == 40 * 2 * 3 * 2 and hits > 40, f"the comparison must exercise real hits ({hits}/{checked})"


# --- partitions: a caller never searches what it may not read -------------------------------------------------

def test_only_the_callers_clearance_partitions_are_ever_searched(db, fake):
    sc.store_cache("ops", question("alpha"), "secret", "the secret answer", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=1)  # identical vector: similarity 1.0
    miss = sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db)
    assert miss == {"hit": False}
    searched = {key[2] for key in cache_index._INDEXES if key[1] == "ops"}
    assert searched == {"public"}, "the secret rows are in no matrix that this lookup touched"
    hit = sc.lookup_cache("ops", question("bravo"), "secret", threshold=0.5, db_path=db)
    assert hit["hit"] and hit["generation"] == "the secret answer"
    assert {key[2] for key in cache_index._INDEXES if key[1] == "ops"} == {"public", "internal", "confidential", "secret"}


def test_other_workspaces_and_index_versions_are_separate_partitions(db, fake, monkeypatch):
    sc.store_cache("ops", question("alpha"), "public", "ops answer", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=2)
    assert sc.lookup_cache("hr", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)["hit"]
    monkeypatch.setattr(sc, "_index_version", lambda ws: "re-indexed")
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}


def test_numbers_negations_and_language_still_veto_a_close_vector(db, fake):
    sc.store_cache("ops", "What is the VAT rate in 2024?", "public", "7.7% in 2024", db_path=db)
    fake.near("What is the VAT rate in 2025?", "What is the VAT rate in 2024?", 0.0, seed=3)
    fake.near("What is the VAT rate in 2024 please?", "What is the VAT rate in 2024?", 0.02, seed=4)
    assert sc.lookup_cache("ops", "What is the VAT rate in 2025?", "public", threshold=0.5, db_path=db) == {"hit": False}
    assert sc.lookup_cache("ops", "What is the VAT rate in 2024 please?", "public", threshold=0.5, db_path=db)["hit"]


# --- the index is never the authority ---------------------------------------------------------------------------

def test_a_row_edited_or_moved_on_disk_leaves_the_matrix_and_never_matches(db, fake):
    sc.store_cache("ops", question("alpha"), "internal", "internal answer", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=5)
    assert sc.lookup_cache("ops", question("bravo"), "internal", threshold=0.5, db_path=db)["hit"]
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE semantic_cache SET clearance = 'public'")  # move it down: the seal binds the clearance
    for clearance in ("public", "internal"):
        sc._l1_clear()
        assert sc.lookup_cache("ops", question("bravo"), clearance, threshold=0.0, db_path=db) == {"hit": False}


def test_a_row_whose_vector_is_swapped_for_another_rows_does_not_open(db, fake):
    sc.store_cache("ops", question("alpha"), "public", "first", db_path=db)
    sc.store_cache("ops", question("charlie"), "public", "second", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=6)
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)["generation"] == "first"
    with sqlite3.connect(db) as conn:
        a, c = [r[0] for r in conn.execute("SELECT query_vector FROM semantic_cache ORDER BY rowid")]
        conn.execute("UPDATE semantic_cache SET query_vector = ? WHERE generation IS NOT NULL AND rowid = 1", (c,))
        conn.execute("UPDATE semantic_cache SET query_vector = ? WHERE rowid = 2", (a,))
    sc._l1_clear()
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}


def test_changes_by_other_writers_are_seen_on_the_next_lookup(db, fake):
    """A second process (or a re-approval) writes to the same file: no restart, no explicit invalidation."""
    sc.store_cache("ops", question("alpha"), "public", "v1", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=7)
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)["generation"] == "v1"
    cache_index_before = cache_index.stats()["entries"]
    sc.store_cache("ops", question("alpha"), "public", "v2", db_path=db)  # replaced in place
    sc._l1_clear()
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)["generation"] == "v2"
    assert cache_index.stats()["entries"] == cache_index_before, "replaced, not duplicated"
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM semantic_cache")
    sc._l1_clear()
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}
    sc.store_cache("ops", question("delta"), "public", "after a wipe", db_path=db)  # rowids start again at 1
    fake.near(question("echo"), question("delta"), 0.0, seed=8)
    assert sc.lookup_cache("ops", question("echo"), "public", threshold=0.5, db_path=db)["generation"] == "after a wipe"


def test_a_review_flag_changed_in_place_is_honoured_without_reopening_the_row(db, fake):
    sc.store_cache("ops", question("alpha"), "public", "auto-approved", db_path=db, reviewed=False)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=9)
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db, include_unreviewed=False) == {"hit": False}
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE semantic_cache SET reviewed = 1")
    got = sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db, include_unreviewed=False)
    assert got["hit"] and got["reviewed"] is True
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE semantic_cache SET reviewed = 0")
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db, include_unreviewed=False) == {"hit": False}


def test_expired_entries_leave_the_index(db, fake):
    sc.store_cache("ops", question("alpha"), "public", "old", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=10)
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)["hit"]
    stale = (datetime.now(UTC) - timedelta(days=400)).isoformat(timespec="seconds")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE semantic_cache SET created_at = ?", (stale,))
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}
    assert cache_index.stats()["entries"] == 0


def test_clearing_the_cache_clears_the_index(db, fake):
    sc.store_cache("ops", question("alpha"), "public", "x", db_path=db)
    fake.near(question("bravo"), question("alpha"), 0.0, seed=12)
    sc.lookup_cache("ops", question("bravo"), "public", threshold=0.5, db_path=db)
    assert cache_index.stats()["entries"] == 1
    sc.clear_cache(db)
    assert cache_index.stats() == {"partitions": 0, "entries": 0, "bytes": 0}


def test_a_failing_embedder_is_a_miss_not_an_error(db, fake, monkeypatch):
    sc.store_cache("ops", question("alpha"), "public", "x", db_path=db)
    monkeypatch.setattr(fake, "embed_query", lambda text: (_ for _ in ()).throw(RuntimeError("model gone")))
    assert sc.lookup_cache("ops", question("bravo"), "public", threshold=0.0, db_path=db) == {"hit": False}


# --- the index itself ------------------------------------------------------------------------------------------

def rows(*items):
    return [(key, rowid, created, reviewed) for key, rowid, created, reviewed in items]


def test_sync_adds_removes_replaces_and_never_retries_a_row_that_failed_to_open():
    index = PartitionIndex("public")
    opened: list[list[str]] = []
    vectors = {"a": unit([1, 0, 0]), "b": unit([0, 1, 0]), "c": unit([0, 0, 1])}

    def load(keys):
        opened.append(list(keys))
        return {k: (vectors[k], ("sig",)) for k in keys if k != "bad"}

    index.sync(rows(("a", 1, "t1", 1), ("b", 2, "t2", 1), ("bad", 3, "t3", 1)), load)
    assert index.keys == ["a", "b"] and index.matrix.shape == (2, 3)
    index.sync(rows(("a", 1, "t1", 1), ("b", 2, "t2", 1), ("bad", 3, "t3", 1)), load)
    assert opened == [["a", "b", "bad"]], "nothing opened again, the bad row included"
    index.sync(rows(("a", 1, "t1", 1), ("c", 4, "t4", 0), ("bad", 3, "t3", 1)), load)
    assert sorted(index.keys) == ["a", "c"] and opened[-1] == ["c"], "only the new row was opened; b is gone"
    assert index.reviewed[index.keys.index("c")] is False
    index.sync(rows(("a", 9, "t1", 1), ("c", 4, "t4", 0)), load)  # same key, new rowid: a replaced row
    assert opened[-1] == ["a"]
    index.sync(rows(("a", 9, "t1", 1), ("c", 4, "t4", 0), ("bad", 3, "t9", 1)), load)  # the bad row changed: try again
    assert opened[-1] == ["bad"]


def test_vectors_of_another_size_or_no_length_are_never_indexed():
    index = PartitionIndex("public")
    loaded = {"a": (unit([1, 0, 0]), ()), "wide": (unit([1, 0, 0, 0]), ())}
    index.sync(rows(("a", 1, "t", 1), ("wide", 2, "t", 1)), lambda keys: {k: loaded[k] for k in keys})
    assert index.keys == ["a"]
    assert unit([]) is None and unit([0, 0, 0]) is None and unit([float("nan"), 1]) is None
    assert index.search(unit([1, 0, 0, 0]), 0.0, (), True) == [], "a query of another size matches nothing"


def test_search_orders_by_similarity_then_recency_and_applies_the_filters():
    index = PartitionIndex("internal")
    v = unit([1, 0])
    near = unit([1, 0.2])
    loaded = {"new": (v, "s"), "old": (v, "s"), "near": (near, "s"), "other": (v, "different"), "raw": (v, "s")}
    index.sync(rows(("new", 1, "2026-02", 1), ("old", 2, "2026-01", 1), ("near", 3, "2026-03", 1),
                    ("other", 4, "2026-04", 1), ("raw", 5, "2026-05", 0)), lambda keys: {k: loaded[k] for k in keys})
    found = index.search(v, 0.9, "s", include_unreviewed=True)
    assert [m.key for m in found] == ["raw", "new", "old", "near"], "ties go to the newer entry; wrong signature is out"
    assert [m.key for m in index.search(v, 0.9, "s", include_unreviewed=False)] == ["new", "old", "near"]
    assert [m.key for m in index.search(v, 0.999, "s", include_unreviewed=True)] == ["raw", "new", "old"]


def test_memory_budget_drops_the_least_recently_used_partition(monkeypatch):
    cache_index.forget()
    for name in ("one", "two", "three"):
        p = cache_index.partition("db", name, "public", "v")
        p.sync(rows((name, 1, "t", 1)), lambda keys, name=name: {k: (unit(np.ones(256)), ()) for k in keys})
    cache_index.partition("db", "one", "public", "v")  # touch: "two" is now the oldest
    monkeypatch.setattr(cache_index, "MAX_INDEX_BYTES", 2 * 256 * 4 + 1)
    cache_index.enforce_budget()
    assert {k[1] for k in cache_index._INDEXES} == {"three", "one"}
    cache_index.forget()


def test_ten_thousand_entries_are_searched_in_milliseconds():
    index = PartitionIndex("public")
    rng = np.random.default_rng(0)
    big = rng.normal(size=(cache_index.MAX_INDEXED_PER_PARTITION, 384)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    index.keys = [f"k{i}" for i in range(len(big))]
    index.rowids = list(range(len(big)))
    index.created = ["t"] * len(big)
    index.reviewed = [True] * len(big)
    index.signatures = [()] * len(big)
    index.matrix = big
    query = big[1234]
    start = time.perf_counter()
    for _ in range(20):
        found = index.search(query, 0.9, (), True)
    elapsed = (time.perf_counter() - start) / 20
    assert found[0].key == "k1234" and abs(found[0].similarity - 1.0) < 1e-5
    assert elapsed < 0.05, f"{elapsed * 1000:.1f} ms per search over 10,000 entries"
