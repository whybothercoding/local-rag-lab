"""Every vector store answers the same questions the same way: ranking, similarity scale, clearance, atomic publish.

The suite is parametrized over the adapters. Chroma always runs; Qdrant (embedded local mode, and a real server when
QDRANT_TEST_URL is set) and pgvector (a real server, PGVECTOR_TEST_URI) join it when available. REQUIRE_QDRANT=1 /
REQUIRE_PGVECTOR=1 (CI) turn a missing one into a failure instead of a skip. Tests against a real service use their own
namespace (a collection prefix, a schema) and remove it afterwards: never the bare `ws_*` names of a real install.
"""
import os
import uuid

import numpy as np
import pytest

from rag_lab.config import settings
from rag_lab.retrieval.embedding_provider import FINGERPRINT_KEY
from rag_lab.storage import vector_store as vs
from rag_lab.storage.vector_store import (
    ChromaStore,
    Hit,
    VectorStore,
    VectorStoreError,
    WorkspaceNotIndexed,
    get_vector_store,
    guard_hits,
)

DIM = 24
FINGERPRINT = "fastembed:test/model"
CLEARANCES = ["public", "internal", "confidential", "secret"]


def unit_cos(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def corpus(workspace: str, n: int = 12, seed: int = 1):
    """Chunks with vectors of norm ~5 (like the real embeddings, which are not unit length) and mixed clearances."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, DIM)) * 5.0
    ids = [f"{workspace}_{i}" for i in range(n)]
    docs = [f"chunk {i} of {workspace}" for i in range(n)]
    metas = [{"source": f"doc{i % 3}.md", "workspace": workspace, "clearance": CLEARANCES[i % 4], "id": ids[i]} for i in range(n)]
    return ids, docs, metas, vectors


def publish(store: VectorStore, workspace: str, n: int = 12, seed: int = 1, fingerprint: str = FINGERPRINT):
    ids, docs, metas, vectors = corpus(workspace, n, seed)
    store.publish(workspace, ids, docs, metas, vectors.tolist(), fingerprint)
    return ids, docs, metas, vectors


# --- the adapters under test -----------------------------------------------------------------------------------

def _unavailable(reason: str, flag: str):
    if os.getenv(flag) == "1":
        pytest.fail(f"{flag}=1 but {reason}")
    pytest.skip(reason)


@pytest.fixture
def chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    return ChromaStore()


@pytest.fixture(params=["chroma", "qdrant-local", "qdrant-server", "pgvector"])
def store(request, tmp_path, monkeypatch):
    kind = request.param
    if kind == "chroma":
        monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
        yield ChromaStore()
        return
    if kind.startswith("qdrant"):
        try:
            from rag_lab.storage.qdrant_store import QdrantStore
        except ImportError:
            _unavailable("qdrant-client is not installed", "REQUIRE_QDRANT")
        prefix = f"pytest_{uuid.uuid4().hex[:8]}_"
        if kind == "qdrant-local":
            adapter = QdrantStore(":memory:", "", prefix)
        else:
            if os.getenv("RAG_OFFLINE") == "1":
                pytest.skip("the offline run blocks TCP connections made through Python sockets, a Qdrant server included")
            url = os.getenv("QDRANT_TEST_URL")
            if not url:
                _unavailable("QDRANT_TEST_URL is not set (start one with `make scale-up SERVICE=qdrant`)", "REQUIRE_QDRANT")
            adapter = QdrantStore(url, os.getenv("QDRANT_TEST_API_KEY", ""), prefix)
            try:
                adapter._client.get_collections()
            except Exception as exc:
                _unavailable(f"Qdrant is not reachable at {url} ({type(exc).__name__})", "REQUIRE_QDRANT")
        yield adapter
        adapter.drop_namespace()
        return
    from rag_lab.config import pgvector_uri_from_env
    from rag_lab.storage.pgvector_store import PgvectorStore

    uri = os.getenv("PGVECTOR_TEST_URI") or pgvector_uri_from_env()
    if not uri:
        _unavailable("no pgvector server: set PGVECTOR_TEST_URI, or POSTGRES_PASSWORD in .env and "
                     "`make scale-up SERVICE=pgvector`", "REQUIRE_PGVECTOR")
    schema = f"pytest_{uuid.uuid4().hex[:8]}"
    try:
        adapter = PgvectorStore(uri, schema)
    except VectorStoreError as exc:
        _unavailable(str(exc), "REQUIRE_PGVECTOR")
    yield adapter
    adapter.drop_namespace()


# --- conformance ---------------------------------------------------------------------------------------------------

def test_an_unindexed_workspace_is_reported_not_guessed(store):
    assert not store.exists("ghost") and "ghost" not in store.workspaces()
    for call in (lambda: store.count("ghost"), lambda: store.collection_metadata("ghost"),
                 lambda: store.query("ghost", [0.1] * DIM, 3, CLEARANCES)):
        with pytest.raises(WorkspaceNotIndexed):
            call()


def test_ensure_creates_an_empty_workspace(store):
    store.ensure("fresh")
    assert store.exists("fresh") and "fresh" in store.workspaces() and store.count("fresh") == 0
    assert store.query("fresh", [0.1] * DIM, 3, CLEARANCES) == []
    store.ensure("fresh")  # idempotent


def test_publish_stores_chunks_text_metadata_and_the_model_stamp(store):
    ids, docs, metas, _ = publish(store, "ops")
    assert store.exists("ops") and store.count("ops") == len(ids)
    assert store.collection_metadata("ops")[FINGERPRINT_KEY] == FINGERPRINT
    hits = store.query("ops", [1.0] * DIM, len(ids), CLEARANCES)
    assert {h.id for h in hits} == set(ids)
    by_id = {h.id: h for h in hits}
    for cid, doc, meta in zip(ids, docs, metas):
        assert by_id[cid].document == doc
        assert by_id[cid].metadata["source"] == meta["source"] and by_id[cid].metadata["clearance"] == meta["clearance"]


def test_similarity_is_cosine_on_chromas_scale_and_ranked_best_first(store):
    """The relevance gates are calibrated on `1 - cosine distance`; the value must agree, not just the order."""
    ids, _, metas, vectors = publish(store, "ops")
    query = np.random.default_rng(7).normal(size=DIM) * 5.0
    hits = store.query("ops", query.tolist(), 5, CLEARANCES)
    expected = sorted(((unit_cos(query, v), cid) for v, cid in zip(vectors, ids)), reverse=True)[:5]
    assert [h.id for h in hits] == [cid for _, cid in expected]
    for hit, (cosine, _) in zip(hits, expected):
        assert abs(hit.similarity - cosine) < 1e-4, (hit.id, hit.similarity, cosine)
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)


def test_the_callers_clearance_limits_what_is_returned(store):
    ids, _, metas, vectors = publish(store, "ops")
    secret_idx = next(i for i, m in enumerate(metas) if m["clearance"] == "secret")
    query = vectors[secret_idx].tolist()  # identical to a secret chunk: it would rank first without the filter
    public = store.query("ops", query, len(ids), ["public"])
    assert public and all(h.metadata["clearance"] == "public" for h in public)
    # The store itself filters, rather than the guard cleaning up afterwards: asking for 3 of the 3 readable chunks
    # returns all 3 even though the nearest chunks overall belong to higher clearances.
    only_public = [i for i, m in enumerate(metas) if m["clearance"] == "public"]
    assert len(store.query("ops", query, len(only_public), ["public"])) == len(only_public)
    assert ids[secret_idx] not in {h.id for h in public}
    internal = store.query("ops", query, len(ids), ["public", "internal"])
    assert {h.metadata["clearance"] for h in internal} <= {"public", "internal"} and len(internal) > len(public)
    everyone = store.query("ops", query, 3, CLEARANCES)
    assert everyone[0].id == ids[secret_idx] and everyone[0].similarity > 0.9999


def test_a_workspace_never_returns_another_workspaces_chunks(store):
    """Same vectors in both: a shared table (pgvector) or a shared prefix must still keep them apart."""
    a_ids, *_ = publish(store, "alpha", seed=3)
    b_ids, *_ = publish(store, "beta", seed=3)
    hits = store.query("alpha", [1.0] * DIM, 50, CLEARANCES)
    assert hits and {h.id for h in hits} <= set(a_ids) and not ({h.id for h in hits} & set(b_ids))
    assert store.workspaces() >= {"alpha", "beta"}


def test_publish_replaces_the_whole_index(store):
    old_ids, *_ = publish(store, "ops", n=12, seed=1)
    new_ids, *_ = publish(store, "ops", n=5, seed=2, fingerprint="fastembed:another/model")
    assert store.count("ops") == 5
    assert {h.id for h in store.query("ops", [1.0] * DIM, 50, CLEARANCES)} == set(new_ids)
    assert set(old_ids) - set(new_ids), "chunks that only the old index had are gone"
    assert store.collection_metadata("ops")[FINGERPRINT_KEY] == "fastembed:another/model"


def test_a_publish_that_fails_leaves_the_old_index_serving(store):
    old_ids, *_ = publish(store, "ops", n=8, seed=1)
    ids, docs, metas, vectors = corpus("ops", 6, seed=9)
    with pytest.raises(Exception):
        store.publish("ops", ids, docs, metas, vectors[:3].tolist(), FINGERPRINT)  # 6 chunks, 3 vectors
    with pytest.raises(Exception):
        store.publish("ops", ids[:2], docs[:2], metas[:2], [[0.1] * DIM, [0.1] * (DIM + 5)], FINGERPRINT)  # mixed sizes
    assert store.count("ops") == 8
    assert {h.id for h in store.query("ops", [1.0] * DIM, 50, CLEARANCES)} == set(old_ids)
    assert store.collection_metadata("ops")[FINGERPRINT_KEY] == FINGERPRINT
    assert not [w for w in store.workspaces() if "__" in w], "no half-built copy is left behind"


def test_read_vectors_returns_each_chunk_text_with_a_vector_pointing_the_same_way(store):
    ids, docs, _, vectors = publish(store, "ops")
    got = dict(store.read_vectors("ops"))
    assert set(got) == set(docs)
    for doc, vector in zip(docs, vectors):
        assert unit_cos(got[doc], vector) > 0.99999  # a store may normalise; the direction is what ranking uses
    with pytest.raises(WorkspaceNotIndexed):
        list(store.read_vectors("ghost"))


def test_delete_removes_the_workspace_and_is_quiet_when_there_is_none(store):
    publish(store, "ops")
    store.delete("ops")
    assert not store.exists("ops") and "ops" not in store.workspaces()
    store.delete("ops")
    store.delete("never-existed")


def test_a_reused_vector_ranks_like_a_fresh_one(store):
    """The ingest path re-publishes vectors it read back: ranking must not change."""
    ids, docs, metas, vectors = publish(store, "ops")
    before = [h.id for h in store.query("ops", [1.0] * DIM, 6, CLEARANCES)]
    reread = dict(store.read_vectors("ops"))
    store.publish("ops", ids, docs, metas, [reread[d] for d in docs], FINGERPRINT)
    assert [h.id for h in store.query("ops", [1.0] * DIM, 6, CLEARANCES)] == before


def test_results_match_chromas_own_output_not_only_numpy(store, chroma):
    """Same corpus in Chroma and in the adapter under test: same chunks, same order, same similarity."""
    if isinstance(store, ChromaStore):
        pytest.skip("this is Chroma")
    publish(store, "ops", n=30, seed=4)
    publish(chroma, "ops", n=30, seed=4)
    rng = np.random.default_rng(11)
    for allowed in (["public"], ["public", "internal"], CLEARANCES):
        for _ in range(5):
            query = (rng.normal(size=DIM) * 5.0).tolist()
            theirs, reference = store.query("ops", query, 5, allowed), chroma.query("ops", query, 5, allowed)
            assert [h.id for h in theirs] == [h.id for h in reference]
            assert all(abs(a.similarity - b.similarity) < 1e-4 for a, b in zip(theirs, reference))
            assert [h.metadata["clearance"] for h in theirs] == [h.metadata["clearance"] for h in reference]


# --- the guard belongs to the base class, not to the adapters ------------------------------------------------------------

class LeakyStore(VectorStore):
    """An adapter whose server-side filter is broken: it returns every chunk it holds, whatever was asked."""

    name = "leaky"

    def __init__(self, hits):
        self._hits = hits

    def workspaces(self): return {"ops"}
    def ensure(self, workspace): pass
    def exists(self, workspace): return True
    def count(self, workspace): return len(self._hits)
    def collection_metadata(self, workspace): return {FINGERPRINT_KEY: FINGERPRINT}
    def read_vectors(self, workspace): return iter(())
    def publish(self, *a, **k): pass
    def delete(self, workspace): pass

    def _search(self, workspace, vector, n, allowed_clearances):
        return list(self._hits)


def _hit(cid, clearance, workspace="ops", similarity=0.99):
    meta = {"source": "x.md", "workspace": workspace}
    if clearance is not None:
        meta["clearance"] = clearance
    return Hit(cid, f"text {cid}", meta, similarity)


def test_the_base_class_guard_stops_a_chunk_even_when_the_adapters_filter_leaks():
    leaky = LeakyStore([
        _hit("pub", "public"), _hit("sec", "secret"), _hit("conf", "Confidential "), _hit("none", None),
        _hit("blank", ""), _hit("junk", "top-secret"), _hit("other-ws", "public", workspace="hr"),
        _hit("upper", "PUBLIC"),
    ])
    assert [h.id for h in leaky.query("ops", [0.0], 10, ["public"])] == ["pub", "upper"]
    assert [h.id for h in leaky.query("ops", [0.0], 10, ["public", "internal", "confidential"])] == ["pub", "conf", "upper"]
    assert leaky._search("ops", [0.0], 10, ["public"]) and len(leaky._search("ops", [0.0], 10, ["public"])) == 8, \
        "the leak is real: without the guard all eight come back"


def test_the_guard_is_applied_by_query_not_left_to_each_adapter(monkeypatch):
    leaky = LeakyStore([_hit("sec", "secret")])
    monkeypatch.setattr(vs, "guard_hits", lambda hits, ws, allowed: list(hits))  # mutate: no guard
    assert [h.id for h in leaky.query("ops", [0.0], 5, ["public"])] == ["sec"], "with the guard removed it leaks"


def test_guard_hits_in_isolation():
    hits = [_hit("a", "public"), _hit("b", "secret")]
    assert guard_hits(hits, "ops", ["public"]) == [hits[0]]
    assert guard_hits(hits, "ops", []) == []


def test_chroma_keeps_its_unfiltered_fallback_for_old_collections_and_the_guard_still_applies(chroma):
    """Parity with the behaviour before the interface existed (tests/test_rbac.py exercises it end to end)."""
    publish(chroma, "ops")
    from unittest import mock

    collection = chroma._collection("ops")
    real = collection.query

    def flaky(**kw):
        if "where" in kw:
            raise RuntimeError("legacy collection without a where index")
        return real(**kw)

    with mock.patch.object(ChromaStore, "_collection", return_value=mock.Mock(query=flaky, metadata=collection.metadata)):
        hits = chroma.query("ops", [1.0] * DIM, 5, ["public"])
    assert hits and all(h.metadata["clearance"] == "public" for h in hits)


# --- the factory ------------------------------------------------------------------------------------------------------------

def test_the_default_store_is_embedded_chroma_and_an_unknown_one_is_refused(monkeypatch):
    assert settings.vector_store == "chroma" and isinstance(get_vector_store(), ChromaStore)
    monkeypatch.setattr(settings, "vector_store", "milvus")
    with pytest.raises(VectorStoreError, match="chroma, qdrant, pgvector"):
        get_vector_store()


# --- configuration errors say what to do ----------------------------------------------------------------------------------

def test_a_missing_qdrant_package_or_url_is_a_labelled_error(monkeypatch):
    import sys

    monkeypatch.setattr(settings, "vector_store", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    monkeypatch.setitem(sys.modules, "qdrant_client", None)  # as if it were not installed
    monkeypatch.delitem(sys.modules, "rag_lab.storage.qdrant_store", raising=False)
    vs._INSTANCES.clear()
    with pytest.raises(VectorStoreError, match="qdrant-client"):
        get_vector_store()


def test_qdrant_without_a_url_and_pgvector_without_a_uri_are_labelled_errors():
    pytest.importorskip("qdrant_client")
    from rag_lab.storage.pgvector_store import PgvectorStore
    from rag_lab.storage.qdrant_store import QdrantStore

    with pytest.raises(VectorStoreError, match="QDRANT_URL"):
        QdrantStore("", "", "")
    with pytest.raises(VectorStoreError, match="PGVECTOR_URI"):
        PgvectorStore("", "public")
    with pytest.raises(VectorStoreError, match="PGVECTOR_SCHEMA"):
        PgvectorStore("postgresql://u:p@localhost/db", "x; DROP TABLE y")


def test_the_pgvector_pool_is_closed_at_exit_so_the_interpreter_finishes_cleanly():
    """Python 3.14 raises PythonFinalizationError when a pool's worker threads are still running at shutdown."""
    import subprocess
    import sys

    import psycopg
    from psycopg import sql

    from rag_lab.config import pgvector_uri_from_env

    uri = os.getenv("PGVECTOR_TEST_URI") or pgvector_uri_from_env()
    if not uri:
        _unavailable("no pgvector server", "REQUIRE_PGVECTOR")
    try:  # a URI derived from .env says nothing about whether the server is running
        psycopg.connect(uri, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        _unavailable(f"pgvector server unreachable ({type(exc).__name__})", "REQUIRE_PGVECTOR")
    schema = f"pytest_exit_{uuid.uuid4().hex[:6]}"
    code = (
        "import os, sys\n"
        "from rag_lab.storage.pgvector_store import PgvectorStore\n"
        # The URI travels in the environment, never on argv: a failing assertion prints argv, and the URI has a password.
        "PgvectorStore(os.environ['RAG_EXIT_TEST_URI'], sys.argv[1]).ensure('ws')\n"  # then the interpreter exits
    )
    try:
        done = subprocess.run([sys.executable, "-c", code, schema], capture_output=True, text=True,
                              env=os.environ | {"PGVECTOR_URI": "", "RAG_EXIT_TEST_URI": uri})
    finally:
        with psycopg.connect(uri, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    assert done.returncode == 0 and "PythonFinalizationError" not in done.stderr, done.stderr[-400:]


def test_the_instance_cache_never_holds_a_key_or_a_connection_string(monkeypatch):
    """Its keys are printed by tracebacks and by pytest's assertion output."""
    pytest.importorskip("qdrant_client")
    monkeypatch.setattr(settings, "vector_store", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", ":memory:")
    monkeypatch.setattr(settings, "qdrant_api_key", "hunter2-not-a-real-key")
    monkeypatch.setattr(vs, "_INSTANCES", {})
    assert get_vector_store().name == "qdrant"
    assert not any("hunter2" in str(part) for key in vs._INSTANCES for part in key)


def test_factory_reset_purge_empties_a_real_store_of_this_installs_workspaces(store, monkeypatch):
    """`make factory-reset` runs rag_lab.ops.purge_external; here against every adapter, real servers included."""
    from rag_lab.ops import purge_external

    publish(store, "alpha", seed=1)
    publish(store, "beta", seed=2)
    monkeypatch.setattr(settings, "vector_store", "qdrant")  # anything but chroma: the external branch
    monkeypatch.setattr(vs, "get_vector_store", lambda: store)
    lines, complete = purge_external.purge()
    assert complete and lines[0].startswith("Purged 2 workspace(s) from ") and "hunter2" not in lines[0]
    assert store.workspaces() == set() and not store.exists("alpha")
