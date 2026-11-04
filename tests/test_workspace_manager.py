from contextlib import closing
from pathlib import Path
import pytest

from rag_lab.config import settings
from rag_lab.storage.workspace_manager import (
    sanitize_workspace_id,
    list_workspaces,
    create_workspace,
    delete_workspace,
    get_workspace_stats,
    ensure_base_dirs,
)


@pytest.fixture(autouse=True)
def isolated_knowledge_base(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "knowledge_base_dir", str(tmp_path / "knowledge_base"))


def test_sanitize_workspace_id():
    assert sanitize_workspace_id("Accounting & Finance") == "accounting_finance"
    assert sanitize_workspace_id("Workspace---2026!!!") == "workspace_2026"
    assert sanitize_workspace_id("   legal   ") == "legal"
    assert sanitize_workspace_id("___") == ""


def test_create_and_list_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))

    ensure_base_dirs()
    create_workspace("accounting")
    create_workspace("legal_ops")

    workspaces = list_workspaces()
    assert "accounting" in workspaces
    assert "legal_ops" in workspaces


def test_workspace_stats_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))

    ensure_base_dirs()
    stats = get_workspace_stats("nonexistent")
    assert stats["files"] == 0
    assert stats["chunks"] == 0


def test_delete_workspace(tmp_path: Path, monkeypatch):
    from rag_lab.security.access import add_user, load_policy, set_grant
    from conftest import TEST_PASSWORD

    policy_path = str(tmp_path / "access_policy.json")
    monkeypatch.setattr(settings, "access_policy_path", policy_path)
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))

    ensure_base_dirs()
    create_workspace("accounting")
    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query", "review"], path=policy_path)
    assert "accounting" in list_workspaces()
    assert "accounting" in load_policy(path=policy_path)["users"]["erin"]["grants"]

    delete_workspace("accounting")
    assert "accounting" not in list_workspaces()
    assert "accounting" not in load_policy(path=policy_path)["users"]["erin"]["grants"]


def test_workspace_manager_cli(tmp_path: Path, monkeypatch, capsys):
    import rag_lab.storage.workspace_manager as workspace_manager
    from conftest import TEST_PASSWORD

    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)

    # Without --as or with non-admin, create/delete is refused
    assert workspace_manager.main(["create", "HR Policy"]) == 1
    assert workspace_manager.main(["--as", "carol", "create", "HR Policy"]) == 1

    assert workspace_manager.main(["--as", "alice", "create", "HR Policy"]) == 0
    assert "hr_policy" in list_workspaces()

    assert workspace_manager.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "hr_policy:" in out

    # Cancelled delete keeps workspace
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert workspace_manager.main(["--as", "alice", "delete", "hr_policy"]) == 0
    assert "hr_policy" in list_workspaces()

    # Confirmed delete (-y) purges workspace
    assert workspace_manager.main(["--as", "alice", "delete", "hr_policy", "-y"]) == 0
    assert "hr_policy" not in list_workspaces()

    # Status check exits 0 when models and audit log are intact
    assert workspace_manager.main(["status"]) == 0
    assert "System Status" in capsys.readouterr().out


def test_delete_workspace_escapes_sql_like_underscore_wildcard(tmp_path: Path, monkeypatch):
    import sqlite3
    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(settings, "db_uri", f"sqlite:///{db_file}")
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))

    with closing(sqlite3.connect(db_file)) as conn, conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        conn.execute("CREATE TABLE writes (thread_id TEXT)")
        conn.executemany(
            "INSERT INTO checkpoints (thread_id) VALUES (?)",
            [("ui-ops_a-111",), ("ui-opsXa-222",)],
        )
        conn.executemany(
            "INSERT INTO writes (thread_id) VALUES (?)",
            [("ui-ops_a-111",), ("ui-opsXa-222",)],
        )
        conn.commit()

    create_workspace("ops_a")
    delete_workspace("ops_a")

    with closing(sqlite3.connect(db_file)) as conn, conn:
        remaining_cp = [r[0] for r in conn.execute("SELECT thread_id FROM checkpoints").fetchall()]
        remaining_wr = [r[0] for r in conn.execute("SELECT thread_id FROM writes").fetchall()]
    assert remaining_cp == ["ui-opsXa-222"]
    assert remaining_wr == ["ui-opsXa-222"]


def test_workspace_stats_counts_nested_supported_files_and_ignores_unsupported(tmp_path: Path, monkeypatch):
    kb_dir = tmp_path / "knowledge_base"
    up_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "knowledge_base_dir", str(kb_dir))
    monkeypatch.setattr(settings, "uploads_dir", str(up_dir))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))

    nested_kb = kb_dir / "engineering" / "architecture" / "2026"
    nested_kb.mkdir(parents=True)
    (nested_kb / "design.md").write_text("# System Design")
    (nested_kb / "diagram.png").write_bytes(b"\x89PNG")
    (nested_kb / ".DS_Store").write_text("ignored")

    nested_up = up_dir / "engineering" / "notes"
    nested_up.mkdir(parents=True)
    (nested_up / "runbook.txt").write_text("Runbook steps")

    stats = get_workspace_stats("engineering")
    assert stats["files"] == 2



# --- make status: provider-aware readiness ---------------------------------------

def _status(capsys, monkeypatch, *, embedding="fastembed", llm="deterministic", pulled=None, native=None):
    from rag_lab.storage.workspace_manager import print_status

    from rag_lab import accel

    # Whether this machine has the accelerator is not what these tests are about; its own tests set it.
    given = native or accel.AccelStatus("preferred", True, version="test")
    monkeypatch.setattr(accel, "status", lambda: given)
    monkeypatch.setattr(settings, "embedding_provider", embedding)
    monkeypatch.setattr(settings, "llm_provider", llm)
    monkeypatch.setattr("rag_lab.pipeline.llm_provider.ollama_models", lambda *a, **k: pulled)
    code = print_status()
    return code, capsys.readouterr().out


def test_status_default_configuration_is_healthy_without_ollama(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch)
    assert code == 0, out
    assert "Embedding provider  : fastembed:" in out
    assert "deterministic in-process fallback" in out
    assert "indexed with" not in out


def test_status_flags_an_offline_ollama_only_when_the_chat_model_needs_it(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch, llm="ollama")
    assert code == 1
    assert "the configuration requires it" in out
    code, out = _status(capsys, monkeypatch, llm="deterministic")
    assert code == 0 and "Ollama embeddings" not in out, "embeddings never need Ollama"


def test_status_flags_a_missing_chat_model(rollback_index, capsys, monkeypatch):
    from rag_lab.config import settings

    code, out = _status(capsys, monkeypatch, llm="auto", pulled=set())
    assert code == 1
    assert f"{settings.chat_model} is not pulled" in out


def test_status_reports_a_workspace_indexed_with_another_model(rollback_index, capsys, monkeypatch):
    """An index built before the Ollama embedding path was removed carries an ollama: fingerprint."""
    monkeypatch.setattr("rag_lab.retrieval.embedding_provider.embedding_fingerprint", lambda *a, **k: "fastembed:other/model")
    code, out = _status(capsys, monkeypatch)
    assert "default: indexed with fastembed:" in out
    assert "make sync WS=default FORCE=1" in out


def test_status_names_the_way_out_of_a_leftover_ollama_embedding_setting(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch, embedding="ollama")
    assert code == 1
    assert "no longer supported" in out and "make sync FORCE=1" in out


def test_status_rejects_an_unknown_embedding_provider(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch, embedding="bogus")
    assert code == 1
    assert "Unknown EMBEDDING_PROVIDER" in out


# --- make status: the native accelerator ----------------------------------------

def test_status_shows_the_accelerator_as_ok_when_it_is_active(rollback_index, capsys, monkeypatch):
    from rag_lab import accel

    code, out = _status(capsys, monkeypatch, native=accel.AccelStatus("preferred", True, version="0.1.0"))
    assert code == 0, out
    assert "[OK] Context sanitizer   : native accelerator active (rag-accel 0.1.0, RAG_ACCEL=preferred)" in out


def test_status_shows_the_python_fallback_as_degraded_and_fails(rollback_index, capsys, monkeypatch):
    from rag_lab import accel

    code, out = _status(
        capsys, monkeypatch, native=accel.AccelStatus("preferred", False, reason="No module named 'rag_accel'")
    )
    assert code == 1
    assert "[!!] Context sanitizer   : DEGRADED, pure-Python fallback (No module named 'rag_accel')" in out
    assert "wheelhouse" in out and "make rust-build" in out
