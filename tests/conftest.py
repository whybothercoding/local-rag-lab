"""Shared test setup.

Set RAG_OFFLINE=1 (`make test-offline`) to block every outbound TCP connection made through Python
sockets, including localhost. Unix domain sockets stay available for local IPC. PostgreSQL (libpq
opens its own sockets in C) is only used in the offline run when it is on a loopback address.
"""
import atexit
import os
import shutil
import socket
import tempfile

import pytest

# Chroma's anonymous telemetry must never run in tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# A PostgreSQL server for the checkpointer tests: POSTGRES_TEST_URI, or a postgres DB_URI inherited from the
# environment (CI). Read before DB_URI is pointed at a throwaway SQLite file below.
_INHERITED_DB_URI = os.environ.get("DB_URI", "")
POSTGRES_TEST_URI = os.environ.get("POSTGRES_TEST_URI") or (
    _INHERITED_DB_URI if _INHERITED_DB_URI.startswith(("postgres://", "postgresql://")) else ""
)
# Audit log, access policy, and signing key for tests live in throwaway files, never in data/.
_scratch_dir = tempfile.mkdtemp(prefix="rag-test-")
atexit.register(shutil.rmtree, _scratch_dir, ignore_errors=True)
os.environ["AUDIT_DB_PATH"] = os.path.join(_scratch_dir, "audit.db")
os.environ["CACHE_DB_PATH"] = os.path.join(_scratch_dir, "cache.db")
os.environ["ACCESS_POLICY_PATH"] = os.path.join(_scratch_dir, "access_policy.json")
os.environ["ACCESS_KEY_PATH"] = os.path.join(_scratch_dir, "access.key")
os.environ["DATA_KEY_PATH"] = os.path.join(_scratch_dir, "data.key")
os.environ["ANCHOR_PATH"] = os.path.join(_scratch_dir, "audit_anchors.jsonl")
os.environ["BACKUP_DIR"] = os.path.join(_scratch_dir, "backups")
os.environ["DB_URI"] = f"sqlite:///{os.path.join(_scratch_dir, 'checkpoints.db')}"
# Indexes and uploads too: no test may write to the repository's chroma_data/, bm25_indices/, or uploads/.
# knowledge_base/ stays real; tests only read it.
os.environ["CHROMA_PATH"] = os.path.join(_scratch_dir, "chroma")
os.environ["BM25_DIR"] = os.path.join(_scratch_dir, "bm25")
os.environ["UPLOADS_DIR"] = os.path.join(_scratch_dir, "uploads")
# Tests never depend on a running model server; the deterministic synthesizer keeps answers reproducible.
os.environ["LLM_PROVIDER"] = "deterministic"

# Only now may project modules load: settings is built from the environment on first import.
from rag_lab.config import settings  # noqa: E402

# Fail fast if anything imported settings before the redirect above: tests would write real keys and data.
for _field in ("audit_db_path", "cache_db_path", "access_policy_path", "access_key_path", "data_key_path",
               "anchor_path", "backup_dir", "chroma_path", "bm25_dir", "uploads_dir"):
    if not str(getattr(settings, _field)).startswith(_scratch_dir):
        raise RuntimeError(f"settings.{_field} was built before the test redirect: {getattr(settings, _field)}")
if not settings.db_uri.startswith(f"sqlite:///{_scratch_dir}"):
    raise RuntimeError(f"settings.db_uri was built before the test redirect: {settings.db_uri}")

TEST_PASSWORD = "correct-horse-battery"


@pytest.fixture(autouse=True)
def _clean_cache_each_test():
    from rag_lab.retrieval.semantic_cache import clear_cache

    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def rollback_index(tmp_path, monkeypatch):
    """A real, indexed workspace named 'default'. The pipeline retrieves from it; nothing is injected."""
    from rag_lab.ingestion.ingest import ingest_workspace

    source = tmp_path / "kb" / "default"
    source.mkdir(parents=True)
    (source / "company_policy.md").write_text("Rollback SLA is strictly under 5 minutes.", encoding="utf-8")

    store = tmp_path / "store"
    monkeypatch.setattr(settings, "chroma_path", str(store / "chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(store / "bm25"))
    monkeypatch.setattr(settings, "uploads_dir", str(store / "uploads"))

    assert ingest_workspace("default", str(source)).ok
    return source


def _postgres_reachable(uri: str) -> bool:
    try:
        from psycopg_pool import ConnectionPool

        with ConnectionPool(conninfo=uri, min_size=1, max_size=1, timeout=1.5, kwargs={"autocommit": True}) as pool:
            with pool.connection() as conn:
                conn.execute("SELECT 1;")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_uri() -> str:
    """The PostgreSQL test server. Skips when there is none, unless REQUIRE_POSTGRES=1 (CI), where it fails.

    The offline run blocks every TCP connection, so it always skips there.
    """
    from urllib.parse import urlsplit

    from rag_lab.config import POSTGRES_DB_URI

    def unavailable(reason: str):
        if os.getenv("REQUIRE_POSTGRES") == "1":
            pytest.fail(f"REQUIRE_POSTGRES=1 but {reason}")
        pytest.skip(reason)

    # Without an explicit server, use the docker-compose one (`make up`) when it is running.
    uri = POSTGRES_TEST_URI or POSTGRES_DB_URI
    if os.getenv("RAG_OFFLINE") == "1" and urlsplit(uri).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.skip("the offline run only uses a PostgreSQL server on this machine")
    if not _postgres_reachable(uri):
        unavailable(f"PostgreSQL not reachable at {uri.split('@')[-1]} (start it with 'make up')")
    return _dedicated_test_database(uri)


def _dedicated_test_database(uri: str) -> str:
    """Point the tests at `<database>_pytest` on the same server, recreated for every test session.

    POSTGRES_TEST_URI is often the app's own database (after `make up`); tests that prune or list every
    thread must never touch the threads a developer is working with. Recreating it matters too: each
    session generates a fresh data key, so threads sealed by an earlier session could not be opened.
    """
    from urllib.parse import urlsplit

    import psycopg
    from psycopg import sql

    parts = urlsplit(uri)
    test_db = f"{(parts.path.lstrip('/') or 'postgres')}_pytest"
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(test_db)))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db)))
    return parts._replace(path=f"/{test_db}").geturl()


def token_for(name: str) -> str:
    """A session token for a test account, as the UI or CLI would issue after a successful sign-in."""
    from rag_lab.security.access import issue_token

    return issue_token(name)


def _seed_access_policy() -> None:
    from rag_lab.security.access import add_user, set_grant, set_user_permissions_everywhere

    add_user("alice", TEST_PASSWORD, admin=True)
    set_user_permissions_everywhere("alice", [])
    set_grant("alice", "default", ["query", "review"])
    set_grant("alice", "ops", ["query", "review"])
    set_grant("alice", "accounting", ["query", "review"])

    add_user("bob", TEST_PASSWORD)
    set_user_permissions_everywhere("bob", [])
    set_grant("bob", "default", ["query", "review"])
    set_grant("bob", "ops", ["query", "review"])

    add_user("carol", TEST_PASSWORD)
    set_user_permissions_everywhere("carol", [])
    set_grant("carol", "default", ["query"])

    add_user("dave", TEST_PASSWORD)
    set_user_permissions_everywhere("dave", [])


_seed_access_policy()

_original_connect = socket.socket.connect


def offline_connect_guard(self, address):
    """socket.connect replacement for the offline run: Unix domain sockets only."""
    if self.family == socket.AF_UNIX:
        return _original_connect(self, address)
    raise ConnectionError(f"Network access is disabled for offline tests (attempted {address}).")


if os.getenv("RAG_OFFLINE") == "1":
    socket.socket.connect = offline_connect_guard
