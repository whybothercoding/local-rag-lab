"""Policy v3: roles, groups, resolution order, the v2 -> v3 migration, and the owner guard."""
import json
import os
import stat
from pathlib import Path

import pytest

import rag_lab.security.access as access
from rag_lab.config import settings
from rag_lab.security.access import (
    PolicyIntegrityError,
    add_to_group,
    add_user,
    authorize,
    create_group,
    delete_group,
    delete_user,
    effective_permissions,
    enable_team_mode,
    get_default_role,
    grant_group,
    growth_suggestions,
    is_admin,
    list_groups,
    load_policy,
    remove_from_group,
    remove_workspace_grants,
    review_mode_report,
    set_grant,
    set_role,
    set_user_permissions_everywhere,
)
from conftest import TEST_PASSWORD

WORKSPACES = ["finance", "ops"]


@pytest.fixture
def policy(tmp_path):
    """An isolated policy whose first administrator is the owner."""
    path = str(tmp_path / "access_policy.json")
    add_user("root", TEST_PASSWORD, admin=True, path=path)
    return path


def perms(path: str, name: str, ws: str) -> list[str]:
    p = load_policy(path)
    return effective_permissions(p, p["users"][name], ws)


# --- Roles and resolution order ---------------------------------------------

def test_fresh_install_owner_and_default_role(policy):
    assert load_policy(policy)["users"]["root"]["role"] == "owner"
    assert get_default_role(policy) == "trusted"
    add_user("ann", TEST_PASSWORD, path=policy)
    assert perms(policy, "ann", "finance") == ["auto_approve", "query"]
    assert not is_admin("ann", policy)


def test_second_admin_is_admin_not_owner(policy):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    assert load_policy(policy)["users"]["bea"]["role"] == "admin"
    assert is_admin("bea", policy)


def test_resolution_order_grant_then_all_then_group_then_role_then_default(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    assert perms(policy, "ann", "finance") == ["auto_approve", "query"]  # 4. default role

    set_role("ann", "member", path=policy)
    assert perms(policy, "ann", "finance") == ["query"]  # 3. own role beats default

    create_group("auditors", path=policy)
    add_to_group("ann", "auditors", path=policy)
    grant_group("auditors", "finance", "reviewer", path=policy)
    assert perms(policy, "ann", "finance") == ["query", "review"]  # 2. group beats own role
    assert perms(policy, "ann", "ops") == ["query"], "a group grant on one workspace stays there"

    set_user_permissions_everywhere("ann", ["query", "auto_approve"], path=policy)
    assert perms(policy, "ann", "finance") == ["auto_approve", "query"]  # 1b. own 'all' grant beats groups

    set_grant("ann", "finance", [], path=policy)
    assert perms(policy, "ann", "finance") == []  # 1. own workspace grant beats everything
    assert not authorize("ann", "finance", "query", path=policy)


def test_group_grants_union_and_any_denial_wins(policy):
    add_user("ann", TEST_PASSWORD, role="none", path=policy)
    for g in ("readers", "approvers", "blocked"):
        create_group(g, path=policy)
        add_to_group("ann", g, path=policy)
    grant_group("readers", "all", "trusted", path=policy)
    grant_group("approvers", "finance", "reviewer", path=policy)
    assert perms(policy, "ann", "finance") == ["auto_approve", "query", "review"], "union across groups"
    assert perms(policy, "ann", "ops") == ["auto_approve", "query"], "an 'all' grant covers other workspaces"
    assert perms(policy, "ann", "future") == ["auto_approve", "query"], "and future ones"

    grant_group("blocked", "finance", "none", path=policy)
    assert perms(policy, "ann", "finance") == [], "a group denial beats every other group"
    assert perms(policy, "ann", "ops") == ["auto_approve", "query"]

    grant_group("readers", "ops", "none", path=policy)
    assert perms(policy, "ann", "ops") == [], "within a group, the workspace grant beats its 'all' grant"

    remove_from_group("ann", "blocked", path=policy)
    grant_group("readers", "ops", "default", path=policy)
    assert perms(policy, "ann", "finance") == ["auto_approve", "query", "review"]
    assert perms(policy, "ann", "ops") == ["auto_approve", "query"]


def test_groups_cannot_confer_administration(policy):
    create_group("ops-team", path=policy)
    with pytest.raises(ValueError, match="Unknown role"):
        grant_group("ops-team", "all", "admin", path=policy)
    with pytest.raises(ValueError, match="Unknown role"):
        access.set_default_role("owner", path=policy)


def test_group_lifecycle_and_workspace_deletion(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    create_group("team", path=policy)
    with pytest.raises(ValueError, match="already exists"):
        create_group("team", path=policy)
    add_to_group("ann", "team", path=policy)
    grant_group("team", "finance", "reviewer", path=policy)
    assert list_groups(policy) == [{"group": "team", "members": ["ann"], "grants": {"finance": "reviewer"}}]

    assert remove_workspace_grants("finance", path=policy) == 1
    assert list_groups(policy)[0]["grants"] == {}

    delete_group("team", path=policy)
    assert list_groups(policy) == []
    assert load_policy(policy)["users"]["ann"]["groups"] == []
    with pytest.raises(ValueError, match="does not exist"):
        add_to_group("ann", "team", path=policy)


def test_group_changes_are_admin_gated(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    for call in (
        lambda: create_group("xx", path=policy, actor="ann"),
        lambda: add_to_group("ann", "xx", path=policy, actor="ann"),
        lambda: grant_group("xx", "all", "trusted", path=policy, actor="ann"),
        lambda: set_role("ann", "admin", path=policy, actor="ann"),
    ):
        with pytest.raises(PermissionError, match="not an administrator"):
            call()


def test_group_changes_are_audited(policy, monkeypatch):
    events = []
    monkeypatch.setattr(access, "audit_record", lambda event, **kw: events.append((event, kw["data"])))
    create_group("team", path=policy, actor="root")
    add_to_group("root", "team", path=policy, actor="root")
    grant_group("team", "finance", "member", path=policy, actor="root")
    set_role("root", "owner", path=policy, actor="root")
    names = [e for e, _ in events]
    assert names == ["group_created", "group_member_added", "group_grant_changed", "role_changed"]
    assert all(data["actor"] == "root" for _, data in events)


# --- Owner guard -------------------------------------------------------------

def test_last_owner_cannot_be_demoted_or_deleted(policy):
    with pytest.raises(ValueError, match="last owner"):
        set_role("root", "admin", path=policy, actor="root")
    with pytest.raises(ValueError, match="last owner"):
        delete_user("root", path=policy, actor="root")
    assert load_policy(policy)["users"]["root"]["role"] == "owner"


def test_only_an_owner_assigns_or_removes_owner(policy):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    add_user("cat", TEST_PASSWORD, path=policy, actor="root")
    with pytest.raises(PermissionError, match="Only an owner"):
        set_role("cat", "owner", path=policy, actor="bea")
    with pytest.raises(PermissionError, match="Only an owner"):
        set_role("root", "member", path=policy, actor="bea")
    with pytest.raises(PermissionError, match="Only an owner"):
        delete_user("root", path=policy, actor="bea")
    with pytest.raises(PermissionError, match="Only an owner"):
        add_user("dan", TEST_PASSWORD, role="owner", path=policy, actor="bea")

    set_role("bea", "owner", path=policy, actor="root")  # a second owner, then the first can step down
    set_role("root", "admin", path=policy, actor="root")
    assert load_policy(policy)["users"]["root"]["role"] == "admin"
    with pytest.raises(ValueError, match="last owner"):
        set_role("bea", "default", path=policy, actor="bea")


# --- Review coverage and growth suggestions ---------------------------------

def test_no_reviewer_gap_is_reported_for_new_accounts(policy):
    set_user_permissions_everywhere("root", [], path=policy)  # an owner who does not read answers
    add_user("ann", TEST_PASSWORD, role="member", path=policy)
    gaps = review_mode_report(WORKSPACES, policy).gaps
    assert any(g.startswith("no reviewer on:") for g in gaps)

    add_user("rev", TEST_PASSWORD, role="reviewer", path=policy)
    gaps = review_mode_report(WORKSPACES, policy).gaps
    assert not any(g.startswith("no reviewer on:") for g in gaps)
    assert any("for rev" in g for g in gaps), "the only reviewer cannot release their own answers"


def test_cli_add_user_warns_about_unreviewable_answers(policy, monkeypatch, capsys):
    monkeypatch.setattr(settings, "access_policy_path", policy)
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)
    monkeypatch.setattr("rag_lab.storage.workspace_manager.list_workspaces", lambda: WORKSPACES)
    set_user_permissions_everywhere("root", [], path=policy)
    assert access.main(["--as", "root", "add-user", "ann", "--role", "member"]) == 0
    out = capsys.readouterr().out
    assert "(member)" in out
    assert "Warning: no reviewer on: finance, ops" in out


def test_cli_add_user_prompts_for_a_role_from_the_second_account(policy, monkeypatch, capsys):
    monkeypatch.setattr(settings, "access_policy_path", policy)
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)
    monkeypatch.setattr("rag_lab.storage.workspace_manager.list_workspaces", lambda: WORKSPACES)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _="": "reviewer")
    assert access.main(["--as", "root", "add-user", "ann"]) == 0
    assert load_policy(policy)["users"]["ann"]["role"] == "reviewer"


def test_cli_roles_and_groups_commands(policy, monkeypatch, capsys):
    monkeypatch.setattr(settings, "access_policy_path", policy)
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)
    monkeypatch.setattr("rag_lab.storage.workspace_manager.list_workspaces", lambda: WORKSPACES)
    add_user("ann", TEST_PASSWORD, path=policy)
    assert access.main(["set-role", "ann", "member"]) == 1, "mutations need --as"
    assert access.main(["--as", "root", "set-role", "ann", "member"]) == 0
    assert access.main(["--as", "root", "create-group", "fin"]) == 0
    assert access.main(["--as", "root", "add-to-group", "ann", "fin"]) == 0
    assert access.main(["--as", "root", "grant-group", "fin", "finance", "--role", "reviewer"]) == 0
    assert access.main(["--as", "root", "set-default", "--role", "member"]) == 0
    assert access.main(["groups"]) == 0
    assert "fin: members ann; grants finance(reviewer)" in capsys.readouterr().out
    assert perms(policy, "ann", "finance") == ["query", "review"]
    assert get_default_role(policy) == "member"
    assert access.main(["--as", "root", "remove-from-group", "ann", "fin"]) == 0
    assert access.main(["--as", "root", "grant", "ann", "ops", "--role", "trusted"]) == 0
    assert perms(policy, "ann", "ops") == ["auto_approve", "query"]
    assert access.main(["list"]) == 0
    assert "ann member" in capsys.readouterr().out


def test_team_mode_moves_trusted_to_member(policy):
    add_user("rev", TEST_PASSWORD, role="reviewer", path=policy)
    enable_team_mode(WORKSPACES, path=policy)
    assert get_default_role(policy) == "member"


def test_growth_suggestions_never_change_the_policy(policy):
    assert growth_suggestions(WORKSPACES, policy) == []
    for name in ("ann", "ben", "cy", "dee"):
        add_user(name, TEST_PASSWORD, path=policy)
    before = Path(policy).read_bytes()
    suggestions = growth_suggestions(WORKSPACES, policy)
    assert any("make team-mode" in s for s in suggestions)
    assert any("second owner" in s for s in suggestions)
    assert not any("groups" in s for s in suggestions)
    add_user("eve", TEST_PASSWORD, path=policy)
    assert any("no groups" in s for s in growth_suggestions(WORKSPACES, policy))
    access.set_default_permissions(["auto_approve"], path=policy)
    assert any("custom permission set" in s for s in growth_suggestions(WORKSPACES, policy))
    before = Path(policy).read_bytes()
    growth_suggestions(WORKSPACES, policy)
    assert Path(policy).read_bytes() == before


# --- v2 -> v3 migration ------------------------------------------------------

def _v2_effective(policy: dict, user: dict, ws: str) -> list[str]:
    """Policy v2's resolution, kept here as the reference the migration must reproduce."""
    grants = user.get("grants") or {}
    if ws in grants:
        return sorted(grants[ws])
    if "default_permissions" in user:
        return sorted(user["default_permissions"])
    return sorted((policy.get("defaults") or {"permissions": []})["permissions"])


def _user(admin: bool, grants: dict | None = None, **extra) -> dict:
    return {"password_hash": "", "admin": admin, "clearance": "public", "grants": grants or {},
            "uid": os.urandom(4).hex(), "token_version": 0, **extra}


def _write_legacy(path: Path, users: dict, defaults: dict | None) -> dict:
    policy: dict = {"version": 2 if defaults is not None else 1, "users": users}
    if defaults is not None:
        policy["defaults"] = defaults
    policy["integrity"] = access._legacy_mac(users, defaults)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True))
    return policy


LEGACY_CASES = {
    "open defaults, admin follows them": (
        {"zed": _user(True), "amy": _user(True, {"ops": ["query", "review"]}), "kim": _user(False)},
        {"permissions": ["auto_approve", "query"]},
    ),
    "conftest-style admin denied everywhere": (
        {"alice": _user(True, {"default": ["query", "review"]}, default_permissions=[]),
         "bob": _user(False, {"ops": ["query", "review"]}, default_permissions=[])},
        {"permissions": ["auto_approve", "query"]},
    ),
    "custom default no role matches": (
        {"root": _user(True), "ann": _user(False, {"finance": []})},
        {"permissions": ["auto_approve"]},
    ),
    "custom user-wide set": (
        {"root": _user(True, default_permissions=["auto_approve", "query", "review"]),
         "ann": _user(False, default_permissions=["review"])},
        {"permissions": ["query"]},
    ),
    "version 1 file stays strict": (
        {"erin": _user(False, {"finance": ["query"]}), "root": _user(True)},
        None,
    ),
}


@pytest.mark.parametrize("case", list(LEGACY_CASES))
def test_migration_preserves_everyones_access(tmp_path, case):
    users, defaults = LEGACY_CASES[case]
    target = tmp_path / "access_policy.json"
    original = _write_legacy(target, json.loads(json.dumps(users)), defaults)
    original_bytes = target.read_bytes()

    migrated = load_policy(str(target))

    assert migrated["version"] == 3
    on_disk = json.loads(target.read_text())
    assert on_disk["version"] == 3 and on_disk["groups"] == {}
    workspaces = {"finance", "ops", "default", "never_seen_before"}
    for user in original["users"].values():
        workspaces |= set(user.get("grants") or {})
    for name, old in original["users"].items():
        new = migrated["users"][name]
        assert "admin" not in new
        assert is_admin(name, str(target)) == bool(old.get("admin")), name
        for ws in sorted(workspaces):
            assert effective_permissions(migrated, new, ws) == _v2_effective(original, old, ws), (case, name, ws)

    admins = sorted(n for n, u in original["users"].items() if u.get("admin"))
    assert [n for n, u in migrated["users"].items() if u.get("role") == "owner"] == admins[:1]

    backups = list(tmp_path.glob("access_policy.json.v*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_bytes, "the backup is the untouched old file"
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600

    load_policy(str(target))
    assert len(list(tmp_path.glob("access_policy.json.v*.bak"))) == 1, "migration runs once"


def test_tampered_v2_policy_is_refused_and_left_untouched(tmp_path):
    target = tmp_path / "access_policy.json"
    policy = _write_legacy(target, {"root": _user(True), "ann": _user(False)}, {"permissions": ["query"]})
    policy["users"]["ann"]["admin"] = True  # promote without re-signing
    target.write_text(json.dumps(policy, indent=2, sort_keys=True))
    tampered = target.read_bytes()

    with pytest.raises(PolicyIntegrityError, match="integrity check"):
        load_policy(str(target))
    assert target.read_bytes() == tampered, "a tampered file must never be re-signed by the migration"
    assert not list(tmp_path.glob("*.bak"))


def test_tamper_detection_survives_migration(tmp_path):
    target = tmp_path / "access_policy.json"
    _write_legacy(target, {"root": _user(True), "ann": _user(False)}, {"permissions": ["query"]})
    load_policy(str(target))

    for mutate in (
        lambda p: p["users"]["ann"].__setitem__("role", "owner"),
        lambda p: p.__setitem__("defaults", {"role": "trusted"}),
        lambda p: p.__setitem__("groups", {"x": {"grants": {"*": "reviewer"}}}),
        lambda p: p.__setitem__("version", 2),
    ):
        policy = json.loads(target.read_text())
        good = target.read_text()
        mutate(policy)
        target.write_text(json.dumps(policy))
        with pytest.raises(PolicyIntegrityError):
            load_policy(str(target))
        target.write_text(good)
    assert load_policy(str(target))["users"]["ann"].get("role") is None


def test_migration_inside_an_update_does_not_deadlock(tmp_path):
    target = tmp_path / "access_policy.json"
    _write_legacy(target, {"root": _user(True), "ann": _user(False)}, {"permissions": ["query"]})
    set_role("ann", "reviewer", path=str(target))  # _update holds the lock and load_policy migrates
    assert load_policy(str(target))["users"]["ann"]["role"] == "reviewer"


def test_migration_is_audited(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(access, "audit_record", lambda event, **kw: events.append((event, kw["data"])))
    target = tmp_path / "access_policy.json"
    _write_legacy(target, {"root": _user(True)}, {"permissions": ["auto_approve", "query"]})
    load_policy(str(target))
    assert events[0][0] == "policy_migrated"
    assert events[0][1]["from"] == 2 and events[0][1]["to"] == 3
    assert events[0][1]["pinned"] == ["root (owner): access pinned to auto_approve/query on all workspaces"]


def test_unknown_future_version_is_refused(tmp_path):
    target = tmp_path / "access_policy.json"
    target.write_text(json.dumps({"version": 4, "users": {}, "integrity": "x"}))
    with pytest.raises(PolicyIntegrityError, match="unknown version"):
        load_policy(str(target))
