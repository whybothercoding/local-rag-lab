"""Operating the optional stores: `make factory-reset` purges them, `make status` asks them, neither shows a credential."""
from pathlib import Path

import pytest

from rag_lab.config import settings
from rag_lab.ops import purge_external
from rag_lab.retrieval import semantic_cache as sc
from rag_lab.storage import vector_store as vs
from rag_lab.storage.vector_store import ChromaStore, safe_location, vector_store_status

ROOT = Path(__file__).resolve().parent.parent


class FakeStore(vs.VectorStore):
    """A store in a dict, to see exactly what purge asks of it."""

    name = "fake"

    def __init__(self, workspaces=("alpha", "beta"), broken=False, stubborn=False):
        self.data = {w: [f"{w} chunk"] for w in workspaces}
        self.broken, self.stubborn = broken, stubborn

    def describe(self):
        return "Fake store at fake.example:6333"

    def workspaces(self):
        if self.broken:
            raise ConnectionError("down")
        return set(self.data)

    def delete(self, workspace):
        if not self.stubborn:
            self.data.pop(workspace, None)

    def ensure(self, workspace): pass
    def exists(self, workspace): return workspace in self.data
    def count(self, workspace): return len(self.data[workspace])
    def collection_metadata(self, workspace): return {}
    def read_vectors(self, workspace): return iter(())
    def publish(self, *a, **k): pass
    def _search(self, *a, **k): return []


@pytest.fixture
def external(monkeypatch):
    """VECTOR_STORE=qdrant (a fake one) so that purge and status have something external to deal with."""
    store = FakeStore()
    monkeypatch.setattr(settings, "vector_store", "qdrant")
    monkeypatch.setattr(vs, "get_vector_store", lambda: store)
    return store


# --- factory-reset -----------------------------------------------------------------------------------------------------

def test_with_only_local_stores_there_is_nothing_external_to_purge():
    lines, complete = purge_external.purge()
    assert complete and lines == ["No external store is configured: nothing to purge beyond the local folders."]
    assert purge_external.main() == 0


def test_the_configured_vector_store_is_emptied_for_this_install(external, capsys):
    lines, complete = purge_external.purge()
    assert complete and external.data == {}
    assert lines == ["Purged 2 workspace(s) from Fake store at fake.example:6333."]
    assert purge_external.main() == 0 and "Purged 0 workspace(s)" in capsys.readouterr().out


def test_an_unreachable_store_is_reported_as_not_purged_and_fails(external, capsys):
    external.broken = True
    lines, complete = purge_external.purge()
    assert not complete and lines[0].startswith("NOT PURGED") and "still on that server" in lines[0]
    assert purge_external.main() == 1 and "NOT PURGED" in capsys.readouterr().out


def test_a_store_that_does_not_actually_delete_is_not_reported_as_purged(external):
    external.stubborn = True
    lines, complete = purge_external.purge()
    assert not complete and "NOT PURGED: 2 workspace(s) remain" in lines[0] and "Purged" not in lines[0]


def test_the_shared_cache_is_emptied_and_an_unreachable_one_is_not_claimed(cache_backend):
    if cache_backend != "postgres":
        pytest.skip("this is about the shared cache")
    sc.store_cache("ops", "Where is the runbook?", "public", "in the wiki", compute_vector=False)
    sc.store_cache("hr", "How many leave days?", "public", "twenty-five", compute_vector=False)
    assert sc.cache_stats()["total_entries"] == 2
    lines, complete = purge_external.purge()
    assert complete and lines[0].startswith("Emptied the shared PostgreSQL") and sc.cache_stats()["total_entries"] == 0


def test_an_unreachable_shared_cache_is_not_claimed_as_purged(monkeypatch):
    monkeypatch.setattr(settings, "cache_backend", "postgres")
    monkeypatch.setattr(sc, "cache_backend_status", lambda: (False, "shared PostgreSQL db:5432/rag is unreachable (OperationalError)"))
    lines, complete = purge_external.purge()
    assert not complete and lines[0].startswith("NOT PURGED") and "cached answers are still there" in lines[0]


def test_factory_reset_purges_the_external_stores_before_it_deletes_the_local_folders():
    recipe = (ROOT / "Makefile").read_text(encoding="utf-8").split("\nfactory-reset:")[1].split("\n\n")[0]
    assert recipe.index("rag_lab.ops.purge_external") < recipe.index("rm -rf chroma_data")
    assert "NOT purged" in recipe and "Qdrant, pgvector or shared-cache" in recipe


# --- make status -------------------------------------------------------------------------------------------------------------

def test_the_vector_store_line_names_the_store_and_the_count_without_a_credential():
    healthy, line = vector_store_status()
    assert healthy and line.startswith("embedded Chroma at ") and "workspace(s)" in line


def test_a_dead_vector_store_is_reported_unhealthy(external):
    external.broken = True
    healthy, line = vector_store_status()
    assert not healthy and "Fake store at fake.example:6333 is unreachable (ConnectionError)" in line


def test_a_misconfigured_store_is_reported_unhealthy(monkeypatch):
    monkeypatch.setattr(settings, "vector_store", "milvus")
    healthy, line = vector_store_status()
    assert not healthy and "VECTOR_STORE must be one of" in line


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")  # the client probes a host that does not exist
def test_locations_never_carry_a_user_or_a_password():
    assert safe_location("postgresql://user:hunter2@db.internal:5433/rag_vectors") == "db.internal:5433/rag_vectors"
    assert safe_location("http://localhost:6333") == "localhost:6333"
    pytest.importorskip("qdrant_client")
    from rag_lab.storage.qdrant_store import QdrantStore

    described = QdrantStore("https://qdrant.internal:6333", "hunter2-key", "t_").describe()
    assert described == "Qdrant at qdrant.internal:6333 (prefix 't_')" and "hunter2" not in described


def test_the_sqlite_cache_reports_itself_healthy():
    healthy, where = sc.cache_backend_status()
    assert healthy and where.startswith("SQLite ")


def test_status_exits_1_when_the_vector_store_is_down(rollback_index, capsys, monkeypatch):
    from rag_lab import accel
    from rag_lab.storage.workspace_manager import print_status

    monkeypatch.setattr(accel, "status", lambda: accel.AccelStatus("preferred", True, version="test"))
    monkeypatch.setattr(vs, "vector_store_status", lambda: (False, "Qdrant at db:6333 is unreachable (ConnectError): dense retrieval is unavailable"))
    monkeypatch.setattr("rag_lab.storage.workspace_manager.vector_store_status", vs.vector_store_status)
    code = print_status()
    out = capsys.readouterr().out
    assert code == 1 and "[!!] Vector store        : Qdrant at db:6333 is unreachable" in out


def test_status_exits_1_when_the_shared_cache_is_down(rollback_index, capsys, monkeypatch):
    from rag_lab import accel
    from rag_lab.storage.workspace_manager import print_status

    monkeypatch.setattr(accel, "status", lambda: accel.AccelStatus("preferred", True, version="test"))
    monkeypatch.setattr(sc, "cache_backend_status", lambda: (False, "shared PostgreSQL db:5432/rag is unreachable (OperationalError): nothing is cached or served from it"))
    code = print_status()
    out = capsys.readouterr().out
    assert code == 1 and "[!!] Semantic cache      : shared PostgreSQL db:5432/rag is unreachable" in out
    assert "[OK] Semantic cache" not in out


def test_status_is_healthy_by_default(rollback_index, capsys, monkeypatch):
    from rag_lab import accel
    from rag_lab.storage.workspace_manager import print_status

    monkeypatch.setattr(accel, "status", lambda: accel.AccelStatus("preferred", True, version="test"))
    code = print_status()
    out = capsys.readouterr().out
    assert code == 0, out
    assert "[OK] Vector store        : embedded Chroma at" in out and "[OK] Semantic cache" in out
    assert isinstance(vs.get_vector_store(), ChromaStore)
