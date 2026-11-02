from pathlib import Path

import pytest

from ingest import ingest_workspace
from workspace_manager import (
    discover_knowledge_base_workspaces,
    list_workspaces,
    sync_all_workspaces,
)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch) -> Path:
    kb = tmp_path / "knowledge_base"
    kb.mkdir()
    monkeypatch.setattr("workspace_manager.KNOWLEDGE_BASE_DIR", str(kb))
    monkeypatch.setattr("workspace_manager.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("workspace_manager.BM25_DIR", str(tmp_path / "bm25"))
    monkeypatch.setattr("workspace_manager.UPLOADS_DIR", str(tmp_path / "uploads"))
    return kb


def test_discovery_accepts_plain_folders_only(storage):
    (storage / "Accounting Team").mkdir()
    (storage / "marketing").mkdir()
    (storage / ".hidden").mkdir()
    (storage / "_drafts").mkdir()
    (storage / "loose.txt").write_text("not a workspace")

    found = discover_knowledge_base_workspaces(str(storage))
    assert [ws for ws, _ in found] == ["accounting_team", "marketing"]


def test_discovery_skips_symlinked_folders(storage, tmp_path):
    outside = tmp_path / "outside_secrets"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not index")
    (storage / "leak").symlink_to(outside, target_is_directory=True)

    assert discover_knowledge_base_workspaces(str(storage)) == []


def test_discovery_keeps_first_of_colliding_ids(storage):
    (storage / "Legal Ops").mkdir()
    (storage / "legal_ops").mkdir()

    found = discover_knowledge_base_workspaces(str(storage))
    assert len(found) == 1
    assert found[0][0] == "legal_ops"


def test_list_workspaces_includes_knowledge_base_folders(storage):
    (storage / "hr").mkdir()
    assert "hr" in list_workspaces()


def test_ingest_ignores_symlinked_files(storage, tmp_path):
    ws_dir = storage / "ops"
    ws_dir.mkdir()
    (ws_dir / "real.txt").write_text("Incident severity one requires paging within 10 minutes.")
    outside = tmp_path / "outside.txt"
    outside.write_text("Confidential payroll data.")
    (ws_dir / "link.txt").symlink_to(outside)

    result = ingest_workspace("ops", str(ws_dir))
    assert result.ok
    assert result.indexed_files == ["real.txt"]


def test_manifest_invalidates_when_embedding_model_changes(storage, monkeypatch):
    ws_dir = storage / "ops"
    ws_dir.mkdir()
    (ws_dir / "a.txt").write_text("The deploy window is Tuesday evening.")

    assert ingest_workspace("ops", str(ws_dir)).ok

    monkeypatch.setattr("ingest.embedding_fingerprint", lambda: "fastembed:another-model")
    result = ingest_workspace("ops", str(ws_dir))
    assert result.ok
    assert result.unchanged is False


def test_sync_indexes_each_workspace_then_skips_unchanged(storage):
    (storage / "accounting").mkdir()
    (storage / "accounting" / "tax.txt").write_text("Expense receipts are due within 30 days.")
    (storage / "strategy").mkdir()
    (storage / "strategy" / "roadmap.md").write_text("Q4 focus is reliability and onboarding.")

    first = sync_all_workspaces()
    assert set(first) == {"accounting", "strategy"}
    assert all(r.ok and not r.unchanged for r in first.values())

    second = sync_all_workspaces()
    assert all(r.ok and r.unchanged for r in second.values())
    assert list_workspaces() == ["accounting", "strategy"]


def test_ingest_merges_knowledge_base_and_uploads(storage, tmp_path):
    """Uploading a file via UI (into uploads/<ws>) must merge with knowledge_base/<ws> rather than wiping it."""
    from workspace_manager import get_workspace_stats

    kb_acc = storage / "accounting"
    kb_acc.mkdir()
    (kb_acc / "tax.txt").write_text("Corporate tax policy requires quarterly filings.")

    up_acc = tmp_path / "uploads" / "accounting"
    up_acc.mkdir(parents=True)
    (up_acc / "memo.txt").write_text("Supplemental expense memo for Q4.")

    # Ingest triggered from UI upload dir
    res = ingest_workspace("accounting", str(up_acc))
    assert res.ok
    assert set(res.indexed_files) == {"memo.txt", "tax.txt"}

    stats = get_workspace_stats("accounting")
    assert stats["files"] == 2

