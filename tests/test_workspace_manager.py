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
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))

    ensure_base_dirs()
    create_workspace("accounting")
    assert "accounting" in list_workspaces()

    delete_workspace("accounting")
    assert "accounting" not in list_workspaces()
