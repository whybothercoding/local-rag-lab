"""Lockout recovery: every way an owner or administrator can lose access has an administrator-run fix.

  lost authenticator, has a recovery code ... the user (test_totp)
  lost authenticator and codes, an admin ... an owner            (test_totp, and here for owner -> owner)
  lost authenticator, a sole owner ......... the machine operator (make recover)
  forgot the password, a member/admin ...... any admin / an owner (an owner's: an owner only)
  forgot the password, a sole owner ........ the machine operator (make recover)
"""
import json
import shutil
from pathlib import Path

import pyotp
import pytest

import rag_lab.security.access as access
from rag_lab.config import settings
from rag_lab.security.access import (
    add_user,
    begin_totp_enrollment,
    confirm_totp_enrollment,
    disable_totp,
    load_policy,
    recover_account,
    set_password,
    sign_in,
    totp_enabled,
    verify_token,
)
from rag_lab.security.audit import list_events
from conftest import TEST_PASSWORD

NEW_PASSWORD = "a-brand-new-passphrase-7421"


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = str(tmp_path / "access_policy.json")
    monkeypatch.setattr(settings, "access_policy_path", path)
    add_user("root", TEST_PASSWORD, admin=True, path=path)
    add_user("ann", TEST_PASSWORD, role="member", path=path)
    return path


def enrol(path: str, name: str) -> str:
    enrolment = begin_totp_enrollment(name, path=path, actor=name)
    confirm_totp_enrollment(name, pyotp.TOTP(enrolment.secret).now(), path=path, actor=name)
    return enrolment.secret


def recovery_events() -> list[dict]:
    return list_events(limit=20, event_type="account_recovered_local")


# --- the sole owner, locked out -------------------------------------------------------------------------

def test_a_sole_owner_who_lost_the_authenticator_and_the_codes_is_recovered(policy):
    enrol(policy, "root")
    assert sign_in("root", TEST_PASSWORD, policy).reason == "totp_required", "locked out"
    session = access.issue_token("root", path=policy)

    result = recover_account("root", reset_totp=True, path=policy)

    assert result == {"username": "root", "totp_reset": True, "password_reset": False}
    assert not totp_enabled("root", policy)
    assert sign_in("root", TEST_PASSWORD, policy).ok, "the password still works"
    assert verify_token(session, path=policy) == "", "old sessions end"
    assert load_policy(policy)["users"]["root"]["role"] == "owner" and "ann" in load_policy(policy)["users"]


def test_a_sole_owner_who_forgot_the_password_is_recovered_and_keeps_the_second_factor(policy):
    secret = enrol(policy, "root")
    recover_account("root", new_password=NEW_PASSWORD, path=policy)
    assert sign_in("root", TEST_PASSWORD, policy).reason == "invalid", "the old password is dead"
    assert sign_in("root", NEW_PASSWORD, policy).reason == "totp_required", "only the password was reset"
    assert sign_in("root", NEW_PASSWORD, policy, code=pyotp.TOTP(secret).at(__import__("time").time() + 30)).ok


def test_both_at_once(policy):
    enrol(policy, "root")
    result = recover_account("root", reset_totp=True, new_password=NEW_PASSWORD, path=policy)
    assert result["totp_reset"] and result["password_reset"]
    assert sign_in("root", NEW_PASSWORD, policy).ok


def test_a_pending_unconfirmed_enrolment_is_cleared_too(policy):
    begin_totp_enrollment("root", path=policy, actor="root")
    recover_account("root", reset_totp=True, path=policy)
    assert "totp_pending" not in load_policy(policy)["users"]["root"]


def test_recovery_is_audited_without_secrets(policy):
    enrol(policy, "root")
    recover_account("root", reset_totp=True, new_password=NEW_PASSWORD, path=policy)
    event, = [e for e in recovery_events() if e["data"]["username"] == "root"][:1]
    assert event["data"]["actor"] == "local-recovery"
    assert event["data"]["totp_reset"] and event["data"]["password_reset"] and event["data"]["os_user"]
    assert NEW_PASSWORD not in json.dumps(recovery_events())


def test_a_refused_recovery_changes_nothing(policy):
    before = Path(policy).read_bytes()
    with pytest.raises(ValueError, match="at least|too common|characters"):
        recover_account("root", new_password="short", path=policy)
    with pytest.raises(ValueError, match="Choose what to recover"):
        recover_account("root", path=policy)
    with pytest.raises(ValueError, match="does not exist"):
        recover_account("ghost", reset_totp=True, path=policy)
    assert Path(policy).read_bytes() == before


def test_it_can_be_switched_off(policy, monkeypatch):
    monkeypatch.setattr(settings, "local_recovery", False)
    before = Path(policy).read_bytes()
    with pytest.raises(PermissionError, match="LOCAL_RECOVERY=off"):
        recover_account("root", reset_totp=True, path=policy)
    assert Path(policy).read_bytes() == before


# --- the command line -----------------------------------------------------------------------------------

@pytest.fixture
def terminal(monkeypatch):
    """A terminal answering the confirmation prompt and the password prompts."""
    answers = {"confirm": "root", "password": NEW_PASSWORD}
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": answers["confirm"])
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda prompt="": answers["password"])
    return answers


def test_the_command_needs_no_sign_in_and_recovers_the_account(policy, terminal, capsys):
    enrol(policy, "root")
    assert access.main(["recover", "root", "--totp", "--password"]) == 0
    out = capsys.readouterr().out
    assert "Recovered 'root'" in out and "two-factor sign-in is off" in out and "the password was set" in out
    assert "15 minutes" in out
    assert sign_in("root", NEW_PASSWORD, policy).ok


def test_the_command_refuses_without_a_terminal_so_it_cannot_be_scripted_by_accident(policy, monkeypatch, capsys):
    enrol(policy, "root")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert access.main(["recover", "root", "--totp"]) == 1
    assert "terminal" in capsys.readouterr().err
    assert totp_enabled("root", policy)


def test_the_command_needs_the_account_name_typed_back(policy, terminal, capsys):
    enrol(policy, "root")
    terminal["confirm"] = "yes"
    assert access.main(["recover", "root", "--totp"]) == 0
    assert "Not confirmed" in capsys.readouterr().out
    assert totp_enabled("root", policy)


def test_the_command_needs_to_be_told_what_to_recover_and_for_whom(policy, terminal, capsys):
    assert access.main(["recover", "root"]) == 1
    assert "Choose what to recover" in capsys.readouterr().err
    assert access.main(["recover", "ghost", "--totp"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_the_command_respects_the_off_switch(policy, terminal, monkeypatch, capsys):
    monkeypatch.setattr(settings, "local_recovery", False)
    assert access.main(["recover", "root", "--totp"]) == 1
    assert "LOCAL_RECOVERY=off" in capsys.readouterr().err


def test_a_weak_new_password_is_refused_before_anything_changes(policy, terminal, capsys):
    enrol(policy, "root")
    terminal["password"] = "passwordpassword"
    before = Path(policy).read_bytes()
    assert access.main(["recover", "root", "--password"]) == 1
    assert "too common" in capsys.readouterr().err
    assert Path(policy).read_bytes() == before


# --- everyone else: an administrator fixes it -----------------------------------------------------------

def test_an_owner_resets_another_owners_second_factor(policy):
    add_user("bea", TEST_PASSWORD, role="owner", path=policy, actor="root")
    enrol(policy, "bea")
    disable_totp("bea", None, path=policy, actor="root")
    assert not totp_enabled("bea", policy) and sign_in("bea", TEST_PASSWORD, policy).ok


def test_an_admin_who_lost_the_authenticator_is_reset_by_an_owner_and_not_by_a_peer(policy):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    add_user("cy", TEST_PASSWORD, admin=True, path=policy, actor="root")
    enrol(policy, "bea")
    with pytest.raises(PermissionError, match="Only an owner"):
        disable_totp("bea", None, path=policy, actor="cy")
    disable_totp("bea", None, path=policy, actor="root")
    assert not totp_enabled("bea", policy)


def test_an_admin_cannot_reset_an_owners_password_but_an_owner_can(policy):
    """Otherwise any administrator could take over the account the owner rule protects (and sidestep a missing 2FA)."""
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    add_user("bob", TEST_PASSWORD, role="owner", path=policy, actor="root")
    with pytest.raises(PermissionError, match="Only an owner can reset an owner's password"):
        set_password("root", NEW_PASSWORD, path=policy, actor="bea")
    assert sign_in("root", TEST_PASSWORD, policy).ok, "untouched"

    set_password("bob", NEW_PASSWORD, path=policy, actor="root")  # an owner resets another owner
    set_password("bea", NEW_PASSWORD, path=policy, actor="root")  # an owner resets an admin
    set_password("root", "yet-another-passphrase-9931", path=policy, actor="root")  # and their own
    assert sign_in("bob", NEW_PASSWORD, policy).ok and sign_in("bea", NEW_PASSWORD, policy).ok


def test_any_admin_still_resets_a_members_or_a_peer_admins_password(policy):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    add_user("cy", TEST_PASSWORD, admin=True, path=policy, actor="root")
    set_password("ann", NEW_PASSWORD, path=policy, actor="bea")
    set_password("cy", NEW_PASSWORD, path=policy, actor="bea")
    assert sign_in("ann", NEW_PASSWORD, policy).ok and sign_in("cy", NEW_PASSWORD, policy).ok


def test_a_backup_of_the_policy_is_not_a_recovery_plan(policy, tmp_path):
    """Documented why: restoring an old policy brings back deleted accounts. `recover` changes only the account."""
    snapshot = tmp_path / "policy.before"
    shutil.copy(policy, snapshot)
    add_user("late", TEST_PASSWORD, path=policy, actor="root")
    enrol(policy, "root")
    recover_account("root", reset_totp=True, path=policy)
    assert "late" in load_policy(policy)["users"], "later accounts survive a recovery"
