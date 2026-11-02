import os
import shutil
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

    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

    assert workspace_manager.main(["create", "HR Policy"]) == 0
    assert "hr_policy" in list_workspaces()

    assert workspace_manager.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "hr_policy:" in out

    # Cancelled delete keeps workspace
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert workspace_manager.main(["delete", "hr_policy"]) == 0
    assert "hr_policy" in list_workspaces()

    # Confirmed delete (-y) purges workspace
    assert workspace_manager.main(["delete", "hr_policy", "-y"]) == 0
    assert "hr_policy" not in list_workspaces()

    # Status check exits 0 when models and audit log are intact
    assert workspace_manager.main(["status"]) == 0
    assert "System Status" in capsys.readouterr().out
