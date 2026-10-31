"""Shared test setup.

Set RAG_OFFLINE=1 (`make test-offline`) to block every outbound TCP connection,
including localhost. Unix domain sockets stay available for local IPC.
"""
import atexit
import os
import shutil
import socket
import tempfile

import pytest

# Chroma's anonymous telemetry must never run in tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Audit log, access policy, and signing key for tests live in throwaway files, never in data/.
_scratch_dir = tempfile.mkdtemp(prefix="rag-test-")
atexit.register(shutil.rmtree, _scratch_dir, ignore_errors=True)
os.environ["AUDIT_DB_PATH"] = os.path.join(_scratch_dir, "audit.db")
os.environ["ACCESS_POLICY_PATH"] = os.path.join(_scratch_dir, "access_policy.json")
os.environ["ACCESS_KEY_PATH"] = os.path.join(_scratch_dir, "access.key")
os.environ["DATA_KEY_PATH"] = os.path.join(_scratch_dir, "data.key")
os.environ["ANCHOR_PATH"] = os.path.join(_scratch_dir, "audit_anchors.jsonl")
os.environ["BACKUP_DIR"] = os.path.join(_scratch_dir, "backups")
os.environ["DB_URI"] = f"sqlite:///{os.path.join(_scratch_dir, 'checkpoints.db')}"
# Tests never depend on a running model server; the deterministic synthesizer keeps answers reproducible.
os.environ["LLM_PROVIDER"] = "deterministic"

TEST_PASSWORD = "correct-horse-battery"


@pytest.fixture
def rollback_index(tmp_path, monkeypatch):
    """A real, indexed workspace named 'default'. The pipeline retrieves from it; nothing is injected."""
    from ingest import ingest_workspace

    source = tmp_path / "kb" / "default"
    source.mkdir(parents=True)
    (source / "company_policy.md").write_text("Rollback SLA is strictly under 5 minutes.", encoding="utf-8")

    store = tmp_path / "store"
    for target in ("workspace_manager", "app"):
        monkeypatch.setattr(f"{target}.CHROMA_PATH", str(store / "chroma"))
        monkeypatch.setattr(f"{target}.BM25_DIR", str(store / "bm25"))
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(store / "uploads"))

    assert ingest_workspace("default", str(source)).ok
    return source


def token_for(name: str) -> str:
    """A session token for a test account, as the UI or CLI would issue after a successful sign-in."""
    from access import issue_token

    return issue_token(name)


def _seed_access_policy() -> None:
    from access import add_user, set_grant

    add_user("alice", TEST_PASSWORD, admin=True)
    set_grant("alice", "default", ["query", "review"])
    set_grant("alice", "ops", ["query", "review"])
    set_grant("alice", "accounting", ["query", "review"])

    add_user("bob", TEST_PASSWORD)
    set_grant("bob", "default", ["query", "review"])
    set_grant("bob", "ops", ["query", "review"])

    add_user("carol", TEST_PASSWORD)
    set_grant("carol", "default", ["query"])

    add_user("dave", TEST_PASSWORD)


_seed_access_policy()

if os.getenv("RAG_OFFLINE") == "1":
    _original_connect = socket.socket.connect

    def _guarded_connect(self, address):
        if self.family == socket.AF_UNIX:
            return _original_connect(self, address)
        raise ConnectionError(f"Network access is disabled for offline tests (attempted {address}).")

    socket.socket.connect = _guarded_connect
