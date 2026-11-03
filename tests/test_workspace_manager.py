from contextlib import closing
from pathlib import Path
import pytest
from workspace_manager import (
    sanitize_workspace_id,
    list_workspaces,
    create_workspace,
    delete_workspace,
    get_workspace_stats,
    ensure_base_dirs,
)


@pytest.fixture(autouse=True)
def isolated_knowledge_base(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("workspace_manager.KNOWLEDGE_BASE_DIR", str(tmp_path / "knowledge_base"))


def test_sanitize_workspace_id():
    assert sanitize_workspace_id("Accounting & Finance") == "accounting_finance"
    assert sanitize_workspace_id("Workspace---2026!!!") == "workspace_2026"
    assert sanitize_workspace_id("   legal   ") == "legal"
    assert sanitize_workspace_id("___") == ""


def test_create_and_list_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

    ensure_base_dirs()
    create_workspace("accounting")
    create_workspace("legal_ops")

    workspaces = list_workspaces()
    assert "accounting" in workspaces
    assert "legal_ops" in workspaces


def test_workspace_stats_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

    ensure_base_dirs()
    stats = get_workspace_stats("nonexistent")
    assert stats["files"] == 0
    assert stats["chunks"] == 0


def test_delete_workspace(tmp_path: Path, monkeypatch):
    from access import add_user, load_policy, set_grant
    from conftest import TEST_PASSWORD

    policy_path = str(tmp_path / "access_policy.json")
    monkeypatch.setattr("access.ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

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
    import workspace_manager
    from conftest import TEST_PASSWORD

    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("access.getpass.getpass", lambda _="": TEST_PASSWORD)

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
    import workspace_manager

    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(workspace_manager, "DB_URI", f"sqlite:///{db_file}")
    monkeypatch.setattr(workspace_manager, "UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(workspace_manager, "BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr(workspace_manager, "CHROMA_PATH", str(tmp_path / "chroma"))

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
    monkeypatch.setattr("workspace_manager.KNOWLEDGE_BASE_DIR", str(kb_dir))
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(up_dir))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

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

def _status(capsys, monkeypatch, *, embedding="fastembed", llm="deterministic", pulled=None):
    from workspace_manager import print_status

    monkeypatch.setattr("embedding_provider.EMBEDDING_PROVIDER", embedding)
    monkeypatch.setattr("config.LLM_PROVIDER", llm)
    monkeypatch.setattr("llm_provider.ollama_models", lambda *a, **k: pulled)
    code = print_status()
    return code, capsys.readouterr().out


def test_status_default_configuration_is_healthy_without_ollama(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch)
    assert code == 0, out
    assert "Embedding provider  : fastembed:" in out
    assert "deterministic in-process fallback" in out
    assert "indexed with" not in out


def test_status_flags_ollama_embeddings_when_ollama_is_offline(rollback_index, capsys, monkeypatch):
    from config import FASTEMBED_MODEL

    code, out = _status(capsys, monkeypatch, embedding="ollama")
    assert code == 1
    assert "the configuration requires it" in out
    assert f"Model verified      : {FASTEMBED_MODEL}" not in out, "FastEmbed weights are not needed on Ollama"


def test_status_flags_missing_ollama_models(rollback_index, capsys, monkeypatch):
    from config import EMBED_MODEL, OLLAMA_MODEL

    code, out = _status(capsys, monkeypatch, embedding="ollama", llm="auto", pulled=set())
    assert code == 1
    assert f"{OLLAMA_MODEL} is not pulled" in out
    assert f"{EMBED_MODEL} is not pulled" in out


def test_status_reports_a_workspace_indexed_with_another_model(rollback_index, capsys, monkeypatch):
    from config import EMBED_MODEL, OLLAMA_MODEL

    code, out = _status(capsys, monkeypatch, embedding="ollama", llm="ollama", pulled={EMBED_MODEL, OLLAMA_MODEL})
    assert code == 1
    assert "default: indexed with fastembed:" in out
    assert "make sync WS=default FORCE=1" in out


def test_status_rejects_an_unknown_embedding_provider(rollback_index, capsys, monkeypatch):
    code, out = _status(capsys, monkeypatch, embedding="bogus")
    assert code == 1
    assert "Unknown EMBEDDING_PROVIDER" in out
