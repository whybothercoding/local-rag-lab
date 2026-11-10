from pathlib import Path

import pytest

from rag_lab.ingestion.ingest import ingest_workspace
from rag_lab.config import settings
from rag_lab.storage.workspace_manager import (
    discover_knowledge_base_workspaces,
    list_workspaces,
    sync_all_workspaces,
)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch) -> Path:
    kb = tmp_path / "knowledge_base"
    kb.mkdir()
    monkeypatch.setattr(settings, "knowledge_base_dir", str(kb))
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path / "bm25"))
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
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

    monkeypatch.setattr("rag_lab.ingestion.ingest.embedding_fingerprint", lambda: "fastembed:another-model")
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
    from rag_lab.storage.workspace_manager import get_workspace_stats

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


def test_ingest_skips_oversized_empty_and_invalid_clearance_files(storage, monkeypatch):
    monkeypatch.setattr(settings, "max_source_bytes", 100)
    ws_dir = storage / "ops"
    ws_dir.mkdir()
    (ws_dir / "valid.md").write_text("Valid short operational runbook.")
    (ws_dir / "empty.txt").write_text("   \n  ")
    (ws_dir / "oversized.txt").write_text("X" * 250)
    (ws_dir / "bad_clearance.md").write_text("---\nclearance: ultra_secret\n---\nBody text.")

    res = ingest_workspace("ops", str(ws_dir), force=True)
    assert res.ok
    assert res.indexed_files == ["valid.md"]
    assert any("empty.txt: no extractable text" in s for s in res.skipped)
    assert any("oversized.txt: larger than 100 bytes" in s for s in res.skipped)
    assert any("bad_clearance.md: invalid clearance" in s for s in res.skipped)


def test_ingest_produces_identical_chunk_ids_regardless_of_entry_dir(storage, tmp_path):
    import chromadb
    kb_acc = storage / "accounting"
    kb_acc.mkdir()
    (kb_acc / "z_policy.txt").write_text("Corporate tax policy requires quarterly filings.")

    up_acc = tmp_path / "uploads" / "accounting"
    up_acc.mkdir(parents=True)
    (up_acc / "a_memo.txt").write_text("Supplemental expense memo for Q4.")

    assert ingest_workspace("accounting", str(kb_acc), force=True).ok
    client = chromadb.PersistentClient(path=settings.chroma_path)
    ids_from_kb = sorted(client.get_collection("ws_accounting").get()["ids"])

    assert ingest_workspace("accounting", str(up_acc), force=True).ok
    ids_from_up = sorted(client.get_collection("ws_accounting").get()["ids"])

    assert ids_from_kb == ids_from_up


def test_ingest_reclassifies_when_clearance_changes_and_supports_folder_clearance(storage):
    import chromadb
    import rag_lab.storage.workspace_manager as workspace_manager
    ws_dir = storage / "ops"
    ws_dir.mkdir()
    doc = ws_dir / "runbook.md"
    doc.write_text("---\nclearance: public\n---\nRotate credentials every 14 days.")

    res1 = ingest_workspace("ops", str(ws_dir))
    assert res1.ok and not res1.unchanged

    # Changing frontmatter clearance from public -> secret with identical body triggers re-index
    doc.write_text("---\nClearance: Secret\n---\nRotate credentials every 14 days.")
    res2 = ingest_workspace("ops", str(ws_dir))
    assert res2.ok and not res2.unchanged

    client = chromadb.PersistentClient(path=settings.chroma_path)
    metas = client.get_collection("ws_ops").get()["metadatas"]
    assert all(m["clearance"] == "secret" for m in metas)

    # Folder-level .clearance file sets default clearance for unlabelled files in that folder
    sub = ws_dir / "confidential_notes"
    sub.mkdir()
    (sub / ".clearance").write_text("confidential\n")
    (sub / "note.txt").write_text("Executive board meeting schedule.")
    res3 = ingest_workspace("ops", str(ws_dir))
    assert res3.ok and not res3.unchanged
    all_metas = client.get_collection("ws_ops").get()["metadatas"]
    note_metas = [m for m in all_metas if "note.txt" in m["source"]]
    assert note_metas and all(m["clearance"] == "confidential" for m in note_metas)

    # Removing all files and running force=True purges the workspace index
    for f in ws_dir.rglob("*"):
        if f.is_file():
            f.unlink()
    res_empty = ingest_workspace("ops", str(ws_dir), force=True)
    assert res_empty.ok and res_empty.documents == 0
    assert workspace_manager.get_workspace_stats("ops")["chunks"] == 0





def test_chunks_carry_their_heading_path():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_lab.ingestion.ingest import chunk_document

    doc = ("# Travel Policy\nIntro line.\n\n## Flights\n" + "Economy class is the default. " * 12 +
           "\n\n## Hotels\nHotel rates above 180 EUR need review.\n")
    chunks = chunk_document(doc, RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0))
    assert chunks[0] == "# Travel Policy\nIntro line."
    flights = [c for c in chunks if "Economy" in c]
    assert len(flights) > 1 and all(c.startswith("# Travel Policy\n## Flights\n") for c in flights)
    assert chunks[-1] == "# Travel Policy\n## Hotels\nHotel rates above 180 EUR need review."


def test_documents_without_headings_chunk_as_before():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_lab.ingestion.ingest import chunk_document

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    text = "Plain text policy.\nSecond line."
    assert chunk_document(text, splitter) == splitter.split_text(text)


def test_a_new_chunker_version_re_indexes_unchanged_files(storage, monkeypatch):
    ws_dir = storage / "ops"
    ws_dir.mkdir()
    (ws_dir / "a.md").write_text("# Ops\nThe deploy window is Tuesday evening.")
    assert ingest_workspace("ops", str(ws_dir)).ok
    assert ingest_workspace("ops", str(ws_dir)).unchanged is True
    monkeypatch.setattr("rag_lab.ingestion.ingest.CHUNKER_VERSION", "headings-v3")
    assert ingest_workspace("ops", str(ws_dir)).unchanged is False


def test_a_malformed_clearance_marker_fails_closed_for_the_whole_folder(tmp_path):
    """One bad `.clearance` must refuse every file in that folder, not just the first.

    The cache entry may only be written after the marker validates: caching `None` before the raise
    would let sibling files silently fall back to the parent's (or public) clearance."""
    from rag_lab.ingestion.ingest import _folder_default

    folder = tmp_path / "ws" / "docs"
    folder.mkdir(parents=True)
    (folder / ".clearance").write_text("top secret\n", encoding="utf-8")
    cache: dict = {}
    for name in ("a.md", "b.md"):
        with pytest.raises(ValueError, match="top secret"):
            _folder_default(folder / name, tmp_path / "ws", cache)
    assert cache == {}


def test_a_failed_graph_delete_during_a_purge_is_logged_not_swallowed(storage, monkeypatch, caplog):
    """A purge that cannot delete the knowledge graph must say so: silently leaving it behind
    contradicts the rule that degradation is never silent."""
    import logging

    ws_dir = storage / "ops"
    ws_dir.mkdir()
    (ws_dir / "a.md").write_text("# Ops\nThe deploy window is Tuesday evening.")
    assert ingest_workspace("ops", str(ws_dir)).ok
    monkeypatch.setattr(settings, "cache_db_path", str(storage.parent / "cache.db"))

    def boom(ws_id):
        raise RuntimeError("graph store locked")

    monkeypatch.setattr("rag_lab.retrieval.graph_store.delete_workspace_graph", boom)
    with caplog.at_level(logging.WARNING, logger="ingest"):
        import rag_lab.ingestion.ingest as ing
        ing._clear_index("ops")
    assert any("Knowledge graph" in record.message for record in caplog.records)


def test_a_workspace_literally_named_all_is_reserved(storage):
    """"all" is the everywhere-sentinel in grants and the UI: a workspace of that name would be
    indistinguishable from it, so creating it is refused and discovery skips such a folder."""
    from rag_lab.storage.workspace_manager import RESERVED_WORKSPACE_IDS, create_workspace

    assert create_workspace("all") == ""
    (storage / "all").mkdir()
    assert all(ws_id != "all" for ws_id, _path in discover_knowledge_base_workspaces())
    assert "all" in RESERVED_WORKSPACE_IDS
