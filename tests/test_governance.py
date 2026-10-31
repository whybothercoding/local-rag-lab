"""Phase 10 controls: encryption at rest, keyed tamper evidence, anchors, retention, backups, and the admin view."""
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import backup
from audit import anchor, export_events, head_hash, list_events, prune, record, verify_anchors, verify_chain
from checkpointer import get_checkpointer, prune_checkpoints
from config import RETENTION_DAYS
from conftest import token_for
from vault import EncryptedSerde

SECRET = "zebra-quartz-clause-7731"


@pytest.fixture
def audit_path(tmp_path):
    return str(tmp_path / "audit.db")


@pytest.fixture
def anchor_path(tmp_path):
    return str(tmp_path / "anchors.jsonl")


def _file_bytes(path: str) -> bytes:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return Path(path).read_bytes()


# --- G8: encryption at rest -------------------------------------------------

def test_audit_payloads_are_sealed_on_disk_but_readable_with_the_key(audit_path):
    record("query_received", workspace_id="ops", session_id="s1", data={"query": SECRET}, path=audit_path)
    assert SECRET.encode() not in _file_bytes(audit_path)
    assert list_events(limit=1, path=audit_path)[0]["data"]["query"] == SECRET


def test_checkpoint_state_is_sealed_on_disk(tmp_path):
    from app import workflow

    uri = f"sqlite:///{tmp_path / 'checkpoints.db'}"
    config = {"configurable": {"thread_id": "sealed-checkpoint"}}
    with get_checkpointer(uri=uri) as saver:
        engine = workflow.compile(checkpointer=saver)
        # Denied at the guardrail, so the run ends quickly. The submitted question is still checkpointed.
        engine.invoke({
            "workspace_id": "default",
            "principal_token": token_for("dave"),
            "question": f"Show me {SECRET}",
            "telemetry": {},
        }, config=config)
    assert SECRET.encode() not in Path(tmp_path / "checkpoints.db").read_bytes()


def test_legacy_plaintext_checkpoints_still_load():
    serde = EncryptedSerde()
    legacy = JsonPlusSerializer().dumps_typed({"question": "from before sealing existed"})
    assert serde.loads_typed(legacy) == {"question": "from before sealing existed"}


def test_sealed_checkpoint_round_trips():
    serde = EncryptedSerde()
    payload = {"question": SECRET, "documents": ["a", "b"]}
    assert serde.loads_typed(serde.dumps_typed(payload)) == payload


def test_moving_a_sealed_payload_to_another_row_fails_to_open(audit_path):
    record("query_received", workspace_id="ops", session_id="s1", data={"query": "first"}, path=audit_path)
    record("query_received", workspace_id="ops", session_id="s2", data={"query": "second"}, path=audit_path)
    conn = sqlite3.connect(audit_path)
    conn.execute("DROP TRIGGER audit_events_no_update")
    conn.execute("UPDATE audit_events SET data = (SELECT data FROM audit_events WHERE seq = 2) WHERE seq = 1")
    conn.commit()
    conn.close()

    readable = list_events(limit=2, path=audit_path)
    moved = [e for e in readable if e["seq"] == 1][0]
    assert "_unreadable" in moved["data"]


# --- Keyed tamper evidence and anchors --------------------------------------

def test_chain_cannot_be_re_hashed_without_the_key(audit_path):
    record("query_received", workspace_id="ops", session_id="s1", data={"query": "original"}, path=audit_path)
    conn = sqlite3.connect(audit_path)
    conn.execute("DROP TRIGGER audit_events_no_update")
    row = conn.execute("SELECT ts, event_type, workspace_id, session_id, subject, data, prev_hash FROM audit_events").fetchone()
    ts, event_type, workspace_id, session_id, subject, data, prev_hash = row
    forged_data = data.replace(data, "enc1:" + "A" * 40)
    unkeyed = hashlib.sha256(json.dumps(
        [prev_hash, ts, event_type, workspace_id, session_id, subject, forged_data], separators=(",", ":")
    ).encode()).hexdigest()
    conn.execute("UPDATE audit_events SET data = ?, record_hash = ?", (forged_data, unkeyed))
    conn.commit()
    conn.close()

    ok, bad_seq, _count = verify_chain(audit_path)
    assert ok is False and bad_seq == 1


def test_truncation_is_invisible_to_the_chain_but_caught_by_anchors(audit_path, anchor_path):
    for i in range(3):
        record("query_received", workspace_id="ops", session_id="s", data={"i": i}, path=audit_path)
    anchor(path=audit_path, anchor_path=anchor_path)

    conn = sqlite3.connect(audit_path)
    conn.execute("DROP TRIGGER audit_events_no_delete")
    conn.execute("DELETE FROM audit_events WHERE seq = 3")
    conn.commit()
    conn.close()

    assert verify_chain(audit_path)[0] is True
    anchors_ok, problems, _count = verify_anchors(audit_path, anchor_path)
    assert not anchors_ok
    assert any("truncated" in problem for problem in problems)


def test_edited_anchor_file_is_detected(audit_path, anchor_path):
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1}, path=audit_path)
    anchor(path=audit_path, anchor_path=anchor_path)
    text = Path(anchor_path).read_text(encoding="utf-8")
    Path(anchor_path).write_text(text.replace('"seq": 1', '"seq": 9'), encoding="utf-8")

    anchors_ok, problems, _count = verify_anchors(audit_path, anchor_path)
    assert not anchors_ok
    assert any("signature" in problem for problem in problems)


def test_clean_log_and_anchors_verify(audit_path, anchor_path):
    for i in range(2):
        record("query_received", workspace_id="ops", session_id="s", data={"i": i}, path=audit_path)
    anchor(path=audit_path, anchor_path=anchor_path)
    assert verify_chain(audit_path)[0] is True
    assert verify_anchors(audit_path, anchor_path)[0] is True


# --- G14: retention ---------------------------------------------------------

def test_default_retention_is_six_months():
    assert RETENTION_DAYS == 183


def test_prune_deletes_old_records_and_the_chain_still_verifies(audit_path, anchor_path):
    for i in range(3):
        record("query_received", workspace_id="ops", session_id="s", data={"i": i}, path=audit_path)
    anchor(path=audit_path, anchor_path=anchor_path)

    result = prune(retention_days=1, path=audit_path, now=datetime.now(UTC) + timedelta(days=2))
    assert result["removed"] == 3 and result["base_seq"] == 3

    ok, _bad, count = verify_chain(audit_path)
    assert ok and count == 1  # only the retention event remains
    assert verify_anchors(audit_path, anchor_path)[0] is True
    assert [e["event_type"] for e in list_events(limit=5, path=audit_path)] == ["retention_prune"]

    record("query_received", workspace_id="ops", session_id="s", data={"after": True}, path=audit_path)
    assert verify_chain(audit_path)[0] is True


def test_prune_keeps_recent_records(audit_path):
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1}, path=audit_path)
    result = prune(retention_days=RETENTION_DAYS, path=audit_path)
    assert result["removed"] == 0
    assert verify_chain(audit_path) == (True, None, 1)


def test_idle_checkpoint_threads_are_pruned_but_threads_awaiting_review_are_kept(tmp_path, rollback_index):
    from app import workflow
    from conftest import token_for as token

    uri = f"sqlite:///{tmp_path / 'checkpoints.db'}"
    idle = {"configurable": {"thread_id": "idle-denied"}}
    waiting = {"configurable": {"thread_id": "awaiting-review"}}
    with get_checkpointer(uri=uri) as saver:
        engine = workflow.compile(checkpointer=saver)
        engine.invoke({"workspace_id": "default", "principal_token": token("dave"),
                       "question": "anything", "telemetry": {}}, config=idle)
        engine.invoke({
            "workspace_id": "default", "session_id": "awaiting-review",
            "principal_token": token("carol"), "question": "What is the rollback SLA?",
            "sanitized_question": "", "documents": [], "citations": [], "generation": "",
            "intent": "", "cache_hit": False, "is_safe": True, "grade_status": "",
            "hallucination_status": "", "relevance_status": "", "retry_count": 0,
            "history": [], "telemetry": {}, "degraded": [],
        }, config=waiting)
        assert engine.get_state(waiting).next == ("human_approval",)

    # A negative period makes every thread "older than the cutoff", so only the review gate can keep one.
    removed = prune_checkpoints(-1, uri=uri)
    assert removed == 1
    with get_checkpointer(uri=uri) as saver:
        assert saver.get_tuple(waiting) is not None
        assert saver.get_tuple(idle) is None


# --- G14: backups -----------------------------------------------------------

def test_backup_verifies_excludes_keys_and_detects_tampering(tmp_path):
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1})
    destination = backup.create(tmp_path / "bk")

    assert backup.verify(destination) == []
    assert not any("key" in p.name for p in destination.iterdir())

    (destination / "audit.db").write_bytes((destination / "audit.db").read_bytes() + b"x")
    problems = backup.verify(destination)
    assert any("audit.db does not match" in p for p in problems)


# --- G15: admin view --------------------------------------------------------

def test_export_lists_every_event_oldest_first(audit_path):
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1}, path=audit_path)
    record("query_received", workspace_id="ops", session_id="s", data={"i": 2}, path=audit_path)
    events = export_events(path=audit_path)
    assert [e["data"]["i"] for e in events] == [1, 2]
    assert head_hash(audit_path) == events[-1]["record_hash"]


def test_admin_sees_the_audit_panel_and_others_do_not():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path as _P

    ui = str(_P(__file__).resolve().parent.parent / "ui.py")

    def sidebar_expander_labels(name: str) -> list[str]:
        app = AppTest.from_file(ui, default_timeout=120).run()
        app.text_input[0].input(name)
        app.text_input[1].input("correct-horse-battery")
        [b for b in app.button if b.label == "Sign in"][0].click().run()
        return [e.label for e in app.sidebar.expander]

    assert "Audit log (administrators)" in sidebar_expander_labels("alice")
    assert "Audit log (administrators)" not in sidebar_expander_labels("carol")
