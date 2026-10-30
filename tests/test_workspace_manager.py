import os
import shutil
import pytest
from workspace_manager import (
    sanitize_workspace_id,
    create_workspace,
    list_workspaces,
    delete_workspace,
    get_workspace_stats,
    UPLOADS_DIR,
    CHROMA_PATH,
    BM25_DIR,
)

TEST_WS = "test_accounting_bundle"


@pytest.fixture(autouse=True)
def cleanup():
    """Ensures test workspace is purged before and after each test."""
    delete_workspace(TEST_WS)
    yield
    delete_workspace(TEST_WS)


def test_sanitize_workspace_id():
    assert sanitize_workspace_id("Accounting 2026!") == "accounting_2026"
    assert sanitize_workspace_id("  Strategy & Ops - Marketing ") == "strategy_ops_marketing"
    assert sanitize_workspace_id("legal-contracts") == "legal_contracts"
    assert sanitize_workspace_id("???") == ""


def test_create_and_list_workspace():
    ws_id = create_workspace(TEST_WS)
    assert ws_id == TEST_WS

    workspaces = list_workspaces()
    assert TEST_WS in workspaces

    # Verify upload directory created
    assert os.path.exists(os.path.join(UPLOADS_DIR, TEST_WS))


def test_workspace_stats_empty():
    create_workspace(TEST_WS)
    stats = get_workspace_stats(TEST_WS)
    assert stats["workspace_id"] == TEST_WS
    assert stats["files"] == 0
    assert stats["chunks"] == 0
    assert stats["bm25_indexed"] is False


def test_delete_workspace():
    create_workspace(TEST_WS)
    assert TEST_WS in list_workspaces()

    # Create dummy staging file
    dummy_file = os.path.join(UPLOADS_DIR, TEST_WS, "test.txt")
    with open(dummy_file, "w") as f:
        f.write("test content")

    assert os.path.exists(dummy_file)

    # Perform deletion
    result = delete_workspace(TEST_WS)
    assert result is True

    # Assert completely vanished
    assert TEST_WS not in list_workspaces()
    assert not os.path.exists(os.path.join(UPLOADS_DIR, TEST_WS))
