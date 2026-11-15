"""The shipped knowledge base is the single workspace for first use; it must not drift from the project."""
import re
from pathlib import Path

from rag_lab.storage.workspace_manager import discover_knowledge_base_workspaces

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"


def test_one_shipped_workspace():
    found = discover_knowledge_base_workspaces(str(KB))
    assert [ws for ws, _ in found] == ["local_rag_lab"]
    assert len(list((KB / "local_rag_lab").glob("*.md"))) >= 5


def test_every_make_target_it_names_exists():
    phony = set((ROOT / "Makefile").read_text().split(".PHONY:")[1].split("\n")[0].split())
    named = set()
    for doc in (KB / "local_rag_lab").glob("*.md"):
        named |= set(re.findall(r"`make ([a-z][a-z-]*)", doc.read_text()))
    assert named, "the knowledge base should document commands"
    assert named <= phony, sorted(named - phony)
