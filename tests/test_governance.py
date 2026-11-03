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


def test_thread_bound_checkpoint_cannot_be_replayed_across_threads():
    from cryptography.exceptions import InvalidTag
    from vault import bind_thread

    serde = EncryptedSerde(allow_plaintext=False)
    with bind_thread("thread-alpha"):
        sealed = serde.dumps_typed({"principal": "alice", "generation": "approved"})
        assert serde.loads_typed(sealed)["principal"] == "alice"

    # Replaying the exact same ciphertext under a different thread_id fails AES-GCM tag check
    with bind_thread("thread-beta"):
        with pytest.raises(InvalidTag):
            serde.loads_typed(sealed)

    # Unsealed plaintext blob is rejected when allow_plaintext=False
    legacy = JsonPlusSerializer().dumps_typed({"principal": "forged"})
    with pytest.raises(InvalidTag):
        serde.loads_typed(legacy)


def test_invalid_vault_key_length_is_refused(tmp_path):
    from vault import InvalidKeyError, load_or_create_key

    bad_key = tmp_path / "short.key"
    bad_key.write_bytes(b"too-short")
    with pytest.raises(InvalidKeyError, match="expected 32"):
        load_or_create_key(str(bad_key))


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


def test_backup_list_and_audit_export_cli(tmp_path, monkeypatch, capsys):
    import audit

    bk_root = tmp_path / "backups"
    monkeypatch.setattr(backup, "BACKUP_DIR", str(bk_root))
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1})
    backup.create(bk_root / "backup-20261102T000000Z")

    assert backup.main(["list"]) == 0
    assert "backup-20261102T000000Z" in capsys.readouterr().out

    export_file = tmp_path / "audit-out.jsonl"
    assert audit.main(["export", "-o", str(export_file)]) == 0
    assert export_file.is_file()
    assert len(export_file.read_text(encoding="utf-8").strip().splitlines()) >= 1


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


def test_verify_anchors_detects_malformed_lines_broken_chain_and_missing_fields(audit_path, anchor_path):
    from vault import anchor_mac
    from audit import _canonical

    record("query_received", workspace_id="ops", session_id="s", data={"i": 1}, path=audit_path)
    first = anchor(path=audit_path, anchor_path=anchor_path)
    assert first is not None

    # Append invalid JSON, non-object JSON, broken prev chain, and missing seq/hash fields
    bad_prev_body = {"seq": 1, "hash": first["hash"], "ts": "2026-01-01T00:00:00Z", "prev": "wrong_prev"}
    bad_prev_entry = {**bad_prev_body, "mac": anchor_mac(_canonical(bad_prev_body))}

    missing_seq_body = {"ts": "2026-01-01T00:00:01Z", "prev": bad_prev_entry["mac"]}
    missing_seq_entry = {**missing_seq_body, "mac": anchor_mac(_canonical(missing_seq_body))}

    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("{not-valid-json\n")
        handle.write("[1, 2, 3]\n")
        handle.write(json.dumps(bad_prev_entry, sort_keys=True) + "\n")
        handle.write(json.dumps(missing_seq_entry, sort_keys=True) + "\n")

    ok, problems, count = verify_anchors(audit_path, anchor_path)
    assert not ok
    assert count == 3
    assert any("invalid JSON" in p for p in problems)
    assert any("not a JSON object" in p for p in problems)
    assert any("chaining to the previous anchor is broken" in p for p in problems)
    assert any("missing or invalid seq/hash fields" in p for p in problems)


def test_audit_cli_subcommands(capsys):
    import audit

    # 1. anchor on empty log or populated log, head, tail, verify, feedback
    record(
        "hitl_decision",
        workspace_id="default",
        session_id="s-rej",
        data={
            "released": False,
            "approver": "bob",
            "reason": "Citation mismatch",
            "groundedness_ratio": "0.50",
            "degraded": [],
            "citations": ["doc.md (chunk 0)"],
        },
    )
    assert audit.main(["anchor"]) == 0
    assert "Anchored seq" in capsys.readouterr().out

    assert audit.main(["head"]) == 0
    assert len(capsys.readouterr().out.strip()) == 64

    assert audit.main(["verify"]) == 0
    assert "Chain OK" in capsys.readouterr().out

    assert audit.main(["tail", "-n", "5", "--workspace", "default"]) == 0
    assert "hitl_decision" in capsys.readouterr().out

    assert audit.main(["feedback", "-n", "5", "--workspace", "default"]) == 0
    fb_out = capsys.readouterr().out
    assert "Citation mismatch" in fb_out
    assert "approver=bob" in fb_out


def test_backup_verify_edge_cases_and_cli(tmp_path, monkeypatch, capsys):
    bk_root = tmp_path / "backups"
    monkeypatch.setattr(backup, "BACKUP_DIR", str(bk_root))
    record("query_received", workspace_id="ops", session_id="s", data={"i": 1})

    # 1. CLI create & verify clean backup
    assert backup.main(["create"]) == 0
    created_dirs = list(bk_root.iterdir())
    assert len(created_dirs) == 1
    dest = created_dirs[0]
    assert backup.main(["verify", str(dest)]) == 0

    # 2. Missing manifest
    empty_dir = tmp_path / "empty_bk"
    empty_dir.mkdir()
    assert any("no manifest.json" in p for p in backup.verify(empty_dir))

    # 3. Corrupted JSON manifest
    bad_json_dir = tmp_path / "bad_json_bk"
    bad_json_dir.mkdir()
    (bad_json_dir / "manifest.json").write_text("{corrupt", encoding="utf-8")
    assert any("corrupted manifest.json" in p for p in backup.verify(bad_json_dir))

    # 4. Malformed manifest (not a dict with files dict)
    malformed_dir = tmp_path / "malformed_bk"
    malformed_dir.mkdir()
    (malformed_dir / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    assert any("malformed manifest.json" in p for p in backup.verify(malformed_dir))

    # 5. Tampered manifest metadata (audit_records, audit_head, missing audit.db entry)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_records"] = 9999
    manifest["audit_head"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = backup.verify(dest)
    assert any("audit record count does not match" in p for p in problems)
    assert any("audit head hash does not match" in p for p in problems)
    assert backup.main(["verify", str(dest)]) == 1

    del manifest["files"]["audit.db"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("audit.db is missing from manifest" in p for p in backup.verify(dest))


def test_retention_cli(capsys):
    import retention

    assert retention.main(["--days", "0"]) == 1
    assert "at least one day" in capsys.readouterr().err

    assert retention.main(["--days", "30"]) == 0
    out = capsys.readouterr().out
    assert "Audit log: removed" in out
    assert "Checkpoints: removed" in out
    assert "Cache: removed" in out


def test_ui_sign_out_session_expiry_and_review_flows(rollback_index):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path as _P
    import time

    ui = str(_P(__file__).resolve().parent.parent / "ui.py")

    # 1. Wrong password shows error & reset hint
    app = AppTest.from_file(ui, default_timeout=120).run()
    app.text_input[0].input("carol")
    app.text_input[1].input("wrong-password-12345")
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    assert any("Sign-in failed" in e.value for e in app.error)

    # 2. Sign in as carol, populate session state, and verify sign out clears chat state
    app.text_input[0].input("carol")
    app.text_input[1].input("correct-horse-battery")
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    assert app.session_state["user"] == "carol"

    # Simulate idle timeout expiry (>30m ago)
    app.session_state["last_seen"] = time.time() - 3600
    app.run()
    assert "user" not in app.session_state
    assert any("30 minutes of inactivity" in i.value for i in app.info)

    # 3. Sign in as carol, submit a query that pauses at human_approval, verify Separation of Duties,
    #    then sign out and sign in as bob (reviewer) to approve from the review queue
    app.text_input[0].input("carol")
    app.text_input[1].input("correct-horse-battery")
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    carol_thread = app.session_state["thread_id"]
    app.chat_input[0].set_value("What is the rollback SLA?").run()

    # Carol cannot approve her own query (has no review grant either)
    assert any("Candidate answer paused" in w.value for w in app.warning)
    assert not [b for b in app.button if b.label == "Approve Output"]

    # Sign out clears session state keys
    [b for b in app.sidebar.button if b.label == "Sign out"][0].click().run()
    assert "user" not in app.session_state
    assert "messages" not in app.session_state
    assert "thread_id" not in app.session_state

    # Sign in as bob (reviewer on default workspace) and approve carol's paused query from review queue
    app.text_input[0].input("bob")
    app.text_input[1].input("correct-horse-battery")
    [b for b in app.button if b.label == "Sign in"][0].click().run()

    approve_btns = [b for b in app.button if b.label == "Approve Output"]
    assert len(approve_btns) == 1
    approve_btns[0].click().run()

    # Verify carol's thread completed in checkpoint store
    from checkpointer import get_checkpointer
    from app import workflow as wf

    with get_checkpointer() as saver:
        compiled = wf.compile(checkpointer=saver)
        st_after = compiled.get_state({"configurable": {"thread_id": carol_thread}})
        assert st_after.next == ()
        assert st_after.values.get("generation")

