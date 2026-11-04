"""The shared PostgreSQL cache (CACHE_BACKEND=postgres): what differs from the SQLite cache.

The behavioural suites (tests/test_semantic_cache.py, tests/test_cache_index.py) run against both backends through
the `cache_backend` fixture, including the oracle comparison and the tamper tests. This file covers what only exists
for the shared backend.
"""
import re
from pathlib import Path

import pytest

import rag_lab.retrieval.semantic_cache as sc
from rag_lab.config import settings
from rag_lab.retrieval import cache_index

SOURCE = Path(sc.__file__).read_text(encoding="utf-8")


def test_sqlite_only_sql_stays_in_the_sqlite_constants():
    """A statement that only SQLite understands, outside its constant, fails silently on PostgreSQL (the cache's broad
    exception handling swallows it): the dict-style proxy once did exactly that."""
    assert SOURCE.count("INSERT OR REPLACE") == 1 and "_UPSERT_SQLITE = f\"INSERT OR REPLACE" in SOURCE
    assert "PRAGMA" not in SOURCE.split("def _open_db")[1].split("yield conn")[0].split("if _shared():")[1].split("target =")[0]
    statements = re.findall(r'"((?:SELECT|UPDATE|DELETE|INSERT)[^"]*)"', SOURCE)
    assert statements and not [s for s in statements if "%" in s], "a literal % would be read as a placeholder"


def test_the_postgres_pool_is_never_touched_while_the_backend_is_sqlite(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("the SQLite backend must not open a PostgreSQL pool")

    monkeypatch.setattr(sc, "_pg_pool", boom)
    assert settings.cache_backend == "sqlite"
    sc.store_cache("ops", "Where is the runbook?", "public", "in the wiki", db_path=str(tmp_path / "c.db"), compute_vector=False)
    assert sc.lookup_cache("ops", "Where is the runbook?", "public", db_path=str(tmp_path / "c.db"))["hit"]


def test_the_pool_registry_never_holds_a_connection_string(monkeypatch):
    """A dict key is printed by tracebacks and by pytest's assertion output: it must not carry a password."""
    uri = "postgresql://user:hunter2-not-a-real-password@localhost:5432/db"
    monkeypatch.setattr(settings, "cache_backend", "postgres")
    monkeypatch.setattr(settings, "cache_db_uri", uri)

    class FakePool:
        def __init__(self, **kw):
            self.closed = False

        def connection(self):
            raise RuntimeError("never reached: the test only inspects the registry key")

        def close(self):
            pass

    monkeypatch.setattr("psycopg_pool.ConnectionPool", FakePool)
    monkeypatch.setattr(sc, "_PG_POOLS", {})
    monkeypatch.setattr(sc, "_PG_READY", set())
    with pytest.raises(RuntimeError, match="never reached"):
        sc._pg_pool()
    assert sc._PG_POOLS and not any("hunter2" in k or "postgresql" in k for k in sc._PG_POOLS)


def test_a_missing_uri_is_a_labelled_error(monkeypatch):
    monkeypatch.setattr(settings, "cache_backend", "postgres")
    monkeypatch.setattr(settings, "cache_db_uri", "")
    with pytest.raises(RuntimeError, match="CACHE_DB_URI"):
        with sc._open_db():
            pass


def test_the_index_is_named_by_server_and_database_never_by_password(monkeypatch):
    monkeypatch.setattr(settings, "cache_backend", "postgres")
    monkeypatch.setattr(settings, "cache_db_uri", "postgresql://user:s3cret@db.internal:6543/rag_cache")
    identity = sc._db_identity(None)
    assert identity == "postgres://db.internal:6543/rag_cache" and "s3cret" not in identity


def test_l1_is_off_in_shared_mode(monkeypatch):
    sc._l1_clear()
    monkeypatch.setattr(settings, "cache_backend", "postgres")
    sc._l1_put("k", "plaintext answer", ["c"])
    assert sc._l1_get("k") == (False, "", []), "another process's prune or delete could not evict a local copy"
    monkeypatch.setattr(settings, "cache_backend", "sqlite")
    sc._l1_put("k", "plaintext answer", ["c"])
    assert sc._l1_get("k")[0] is True
    sc._l1_clear()


# --- against a real PostgreSQL (skips like the checkpointer tests; REQUIRE_POSTGRES=1 makes it fail instead) ------------

@pytest.fixture
def shared(cache_backend, request):
    if cache_backend != "postgres":
        pytest.skip("this test is about the shared backend")
    return cache_backend


def test_the_table_holds_only_sealed_content_and_a_replaced_row_gets_a_new_serial(shared):
    sc.store_cache("ops", "What is the Omega code?", "secret", "Project-Omega-998811", ["omega.md"])
    with sc._open_db() as conn:
        first = conn.execute("SELECT rowid, query, generation, citations_json, query_vector FROM semantic_cache").fetchall()
    assert len(first) == 1
    held = b"".join(v if isinstance(v, bytes) else str(v).encode() for v in first[0][1:] if v is not None)
    assert b"Omega" not in held and b"omega.md" not in held
    assert isinstance(first[0][4], bytes), "BYTEA comes back as bytes, not memoryview"

    sc.store_cache("ops", "What is the Omega code?", "secret", "Project-Omega-v2", ["omega.md"])
    with sc._open_db() as conn:
        second = conn.execute("SELECT rowid FROM semantic_cache").fetchall()
    assert len(second) == 1 and second[0][0] > first[0][0], "the index detects a replacement by its new rowid"


def test_two_processes_share_one_cache_without_a_restart(shared):
    """A second process is simulated by a second, independent index registry reading the same table."""
    sc.store_cache("ops", "Which rule covers expense limits?", "public", "answer from process A", ["a.md"])
    sc._l1_clear()
    cache_index.forget()  # process B starts with nothing in memory
    hit = sc.lookup_cache("ops", "Which rule covers expense limits?", "public")
    assert hit["hit"] and hit["generation"] == "answer from process A" and hit["hit_type"] == "exact"
    sc.store_cache("ops", "Which rule covers travel limits?", "public", "answer written by B", ["b.md"])
    cache_index.forget()  # and A, restarted, sees B's write
    again = sc.lookup_cache("ops", "Which rule covers travel limits?", "public")
    assert again["hit"] and again["generation"] == "answer written by B"


def test_clearing_one_workspace_is_visible_to_every_process(shared):
    sc.store_cache("ops", "Where is the runbook?", "public", "in the wiki", ["w.md"])
    sc.store_cache("hr", "How many leave days?", "public", "twenty-five", ["h.md"])
    sc.clear_workspace_cache("ops")
    assert sc.lookup_cache("ops", "Where is the runbook?", "public") == {"hit": False}
    assert sc.lookup_cache("hr", "How many leave days?", "public")["hit"]
    stats = sc.cache_stats()
    assert stats["total_entries"] == 1 and stats["workspaces"] == {"hr": 1} and stats["bytes"] > 0


def test_concurrent_writers_do_not_lose_rows(shared):
    import threading

    def writer(n):
        for i in range(5):
            sc.store_cache("ops", f"What is concurrent question {chr(97 + n)}{chr(97 + i)}?", "public", f"answer {n}{i}", ["c.md"],
                           compute_vector=False)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sc.cache_stats()["total_entries"] == 30
