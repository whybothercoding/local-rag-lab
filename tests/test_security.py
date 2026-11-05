"""Phase 8 controls: upload hygiene, signed identity, tamper detection, sign-in throttling,
password policy, idle timeout, admin-only indexing, and parser limits."""
import hashlib
import json
import os
import zipfile

import pytest
from langgraph.checkpoint.memory import MemorySaver

import rag_lab.security.access as access
from rag_lab.security.access import (
    PolicyIntegrityError,
    add_user,
    authorize,
    check_password_strength,
    hash_password,
    is_idle_expired,
    issue_token,
    load_policy,
    needs_rehash,
    save_policy,
    sign_in,
    verify_token,
)
from rag_lab.pipeline.graph import workflow
from conftest import TEST_PASSWORD, token_for
from rag_lab.ingestion.parsers import ParseError, extract_text_from_file
from rag_lab.storage.workspace_manager import safe_upload_name, staged_path
from rag_lab.config import settings


# --- G1: upload names -------------------------------------------------------

@pytest.mark.parametrize("raw", ["../../etc/passwd.md", "..\\..\\evil.md", "/abs/path/notes.txt"])
def test_upload_names_are_reduced_to_their_base_name(raw):
    assert safe_upload_name(raw) in {"passwd.md", "evil.md", "notes.txt"}


@pytest.mark.parametrize("raw", ["../../etc/passwd", ".env.md", "notes\x00.md", "run.exe", "", ".."])
def test_unsafe_or_unsupported_upload_names_are_refused(raw):
    with pytest.raises(ValueError):
        safe_upload_name(raw)


def test_staged_upload_stays_inside_the_workspace_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "uploads"))
    target = staged_path("ops", "../../report.md")
    assert target.parent == (tmp_path / "uploads" / "ops").resolve()
    assert target.name == "report.md"


# --- G3: signed identity ----------------------------------------------------

def test_valid_token_yields_its_account():
    assert verify_token(token_for("bob")) == "bob"


def test_forged_signature_is_rejected():
    body, _ = token_for("bob").split(".")
    assert verify_token(f"{body}.AAAAforged") == ""


def test_tampered_payload_is_rejected():
    _, signature = token_for("carol").split(".")
    forged_body = issue_token("alice").split(".")[0]
    assert verify_token(f"{forged_body}.{signature}") == ""


def test_expired_token_is_rejected():
    token = issue_token("bob", now=1_000_000)
    assert verify_token(token, now=1_000_000 + access.TOKEN_TTL_SECONDS + 1) == ""


@pytest.mark.parametrize("garbage", ["", "not-a-token", "a.b.c", None])
def test_malformed_tokens_are_rejected(garbage):
    assert verify_token(garbage) == ""


def test_graph_ignores_a_bare_name_and_a_forged_token():
    config = {"configurable": {"thread_id": "forged-identity"}}
    engine = workflow.compile(checkpointer=MemorySaver())
    result = engine.invoke({
        "workspace_id": "default",
        "question": "What is the Q4 roadmap?",
        "principal": "alice",
        "principal_token": "forged.token",
        "telemetry": {},
    }, config=config)
    assert "ACCESS DENIED" in result["generation"]


# --- G6: policy tamper detection -------------------------------------------

def test_policy_edited_outside_access_py_fails_closed(tmp_path):
    path = str(tmp_path / "policy.json")
    add_user("erin", TEST_PASSWORD, path=path)
    with open(path, encoding="utf-8") as handle:
        policy = json.load(handle)
    policy["users"]["erin"]["grants"]["finance"] = ["query", "review"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(policy, handle)

    with pytest.raises(PolicyIntegrityError):
        load_policy(path)
    with pytest.raises(PolicyIntegrityError):
        authorize("erin", "finance", "query", path=path)

    # Corrupted JSON also raises PolicyIntegrityError (fail-closed)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{corrupted-json")
    with pytest.raises(PolicyIntegrityError):
        load_policy(path)


def test_policy_file_is_written_with_an_integrity_tag(tmp_path):
    path = str(tmp_path / "policy.json")
    add_user("erin", TEST_PASSWORD, path=path)
    with open(path, encoding="utf-8") as handle:
        assert len(json.load(handle)["integrity"]) == 64


# --- G2: sign-in throttling and audit ---------------------------------------

@pytest.fixture
def isolated_sign_in(tmp_path):
    return str(tmp_path / "policy.json"), str(tmp_path / "audit.db")


def test_five_failures_lock_the_account_even_for_the_right_password(isolated_sign_in):
    policy, audit = isolated_sign_in
    add_user("erin", TEST_PASSWORD, path=policy)

    for _ in range(access.LOCKOUT_MAX_FAILURES):
        assert sign_in("erin", "wrong-password-value", policy, audit).reason == "invalid"

    locked = sign_in("erin", TEST_PASSWORD, policy, audit)
    assert not locked.ok and locked.reason == "locked"
    assert "Too many failed sign-ins" in locked.describe()


def test_successful_sign_in_issues_a_token_for_that_account(isolated_sign_in):
    policy, audit = isolated_sign_in
    add_user("erin", TEST_PASSWORD, path=policy)
    result = sign_in("erin", TEST_PASSWORD, policy, audit)
    assert result.ok and verify_token(result.token) == "erin"


def test_lockout_expires_with_the_window(isolated_sign_in, monkeypatch):
    policy, audit = isolated_sign_in
    add_user("erin", TEST_PASSWORD, path=policy)
    for _ in range(access.LOCKOUT_MAX_FAILURES):
        sign_in("erin", "wrong-password-value", policy, audit)

    monkeypatch.setattr(access, "LOCKOUT_WINDOW_SECONDS", 0)
    assert sign_in("erin", TEST_PASSWORD, policy, audit).ok


def test_failed_attempts_are_audited_without_the_password(isolated_sign_in):
    from rag_lab.security.audit import list_events

    policy, audit = isolated_sign_in
    add_user("erin", TEST_PASSWORD, path=policy)
    sign_in("erin", "wrong-password-value", policy, audit)

    event = list_events(limit=1, path=audit)[0]
    assert event["event_type"] == "sign_in_failed"
    assert event["data"] == {"username": "erin"}
    assert "wrong-password-value" not in json.dumps(event)


# --- G4: password hashing and policy ----------------------------------------

def test_new_hashes_use_the_stronger_scrypt_cost():
    assert hash_password(TEST_PASSWORD).split("$")[1] == str(2**17)
    assert not needs_rehash(hash_password(TEST_PASSWORD))


def test_old_hashes_are_upgraded_on_the_next_successful_sign_in(isolated_sign_in):
    policy, audit = isolated_sign_in
    salt = os.urandom(16)
    old_digest = hashlib.scrypt(TEST_PASSWORD.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32,
                                maxmem=256 * 1024 * 1024)
    old_hash = f"scrypt$16384$8$1${salt.hex()}${old_digest.hex()}"
    save_policy({"version": 1, "users": {"erin": {"password_hash": old_hash, "admin": False, "grants": {}}}}, policy)
    assert needs_rehash(old_hash)

    assert sign_in("erin", TEST_PASSWORD, policy, audit).ok
    upgraded = load_policy(policy)["users"]["erin"]["password_hash"]
    assert not needs_rehash(upgraded)


def test_common_passwords_are_refused():
    with pytest.raises(ValueError, match="too common"):
        check_password_strength("passwordpassword")


def test_passwords_containing_the_account_name_are_refused():
    with pytest.raises(ValueError, match="account name"):
        check_password_strength("bob-is-the-best-2026", "bob")


# --- G5: idle timeout -------------------------------------------------------

def test_idle_timeout_is_thirty_minutes():
    assert not is_idle_expired(last_seen=0, now=access.IDLE_TIMEOUT_SECONDS)
    assert is_idle_expired(last_seen=0, now=access.IDLE_TIMEOUT_SECONDS + 1)


# --- G11: indexing is an administrative action ------------------------------

def test_indexing_refuses_a_non_administrator(mocker):
    from rag_lab.ingestion.ingest import authenticate_admin

    mocker.patch("rag_lab.ingestion.ingest.getpass.getpass", return_value=TEST_PASSWORD)
    with pytest.raises(SystemExit, match="not an administrator"):
        authenticate_admin("carol")


def test_indexing_refuses_a_wrong_password(mocker):
    from rag_lab.ingestion.ingest import authenticate_admin

    mocker.patch("rag_lab.ingestion.ingest.getpass.getpass", return_value="wrong-password-value")
    with pytest.raises(SystemExit, match="Indexing refused"):
        authenticate_admin("alice")


def test_indexing_accepts_an_administrator(mocker):
    from rag_lab.ingestion.ingest import authenticate_admin

    mocker.patch("rag_lab.ingestion.ingest.getpass.getpass", return_value=TEST_PASSWORD)
    assert authenticate_admin("alice") == "alice"


# --- G12: document parser limits --------------------------------------------

def test_zip_bomb_style_office_file_is_refused(tmp_path):
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\x00" * (20 * 1024 * 1024))
    with pytest.raises(ParseError, match="compression ratio"):
        extract_text_from_file(bomb)


def test_corrupt_office_file_is_refused_cleanly(tmp_path):
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"this is not a zip archive")
    with pytest.raises(ParseError, match="not a valid Office"):
        extract_text_from_file(broken)


def test_parser_runs_under_a_time_limit(tmp_path, monkeypatch):
    import rag_lab.ingestion.parsers as parsers
    note = tmp_path / "note.csv"  # a format that is still parsed in the sandboxed child (plain text is decoded in-process)
    note.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(parsers, "PARSE_TIMEOUT_SEC", 0.001)
    with pytest.raises(ParseError, match="longer than"):
        extract_text_from_file(note)
