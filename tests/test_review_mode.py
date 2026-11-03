"""Review mode: open vs team release, reviewer coverage by workspace and clearance, and the team-mode switch."""
import pytest

import access
from access import (
    add_user,
    enable_team_mode,
    get_default_permissions,
    review_mode_report,
    set_default_permissions,
    set_grant,
    set_user_permissions_everywhere,
)
from conftest import TEST_PASSWORD

WORKSPACES = ["finance", "ops"]


@pytest.fixture
def policy(tmp_path):
    """An isolated policy: one administrator, fresh-install defaults (query + auto_approve)."""
    path = str(tmp_path / "access_policy.json")
    add_user("root", TEST_PASSWORD, admin=True, path=path)
    set_user_permissions_everywhere("root", [], path=path)
    return path


def test_fresh_install_is_open_mode_without_noise(policy):
    report = review_mode_report(WORKSPACES, policy)
    assert report.mode == "open"
    assert report.accounts == 1
    assert report.warnings == [], "a single administrator with no grants should see no warnings"


def test_open_mode_with_several_accounts_recommends_team_mode(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    add_user("ben", TEST_PASSWORD, path=policy)
    report = review_mode_report(WORKSPACES, policy)
    assert report.mode == "open"
    assert any("open review mode with 3 accounts" in w and "make team-mode" in w for w in report.warnings)


def test_workspaces_without_a_reviewer_are_listed_once(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    report = review_mode_report(WORKSPACES, policy)
    missing = [w for w in report.warnings if w.startswith("no reviewer on:")]
    assert len(missing) == 1
    assert "finance" in missing[0] and "ops" in missing[0]


def test_a_reviewer_cannot_cover_their_own_questions(policy):
    """Separation of duties: the only reviewer still needs someone else to release their answers."""
    add_user("ann", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("ann", ["query", "review"], path=policy)
    report = review_mode_report(WORKSPACES, policy)
    assert any("'finance'" in w and "ann" in w for w in report.warnings)


def test_reviewer_clearance_must_cover_the_requester(policy):
    add_user("spy", TEST_PASSWORD, clearance="secret", path=policy)
    add_user("rev", TEST_PASSWORD, clearance="internal", path=policy)
    add_user("pub", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("rev", ["query", "review"], path=policy)
    set_user_permissions_everywhere("spy", ["query"], path=policy)
    set_user_permissions_everywhere("pub", ["query"], path=policy)

    report = review_mode_report(WORKSPACES, policy)
    uncovered = [w for w in report.warnings if w.startswith("workspace 'finance'")]
    assert len(uncovered) == 1
    assert "spy" in uncovered[0], "an internal reviewer cannot release a secret account's answers"
    assert "pub" not in uncovered[0]
    assert report.reviewers["finance"] == ["rev"]


def test_coverage_is_per_workspace(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    add_user("rev", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("ann", ["query"], path=policy)
    set_grant("rev", "ops", ["query", "review"], path=policy)
    set_grant("rev", "finance", [], path=policy)
    report = review_mode_report(WORKSPACES, policy)
    assert report.reviewers == {"finance": [], "ops": ["rev"]}
    assert any(w.startswith("no reviewer on: finance.") for w in report.warnings)


def test_team_mode_refuses_without_any_reviewer(policy):
    add_user("ann", TEST_PASSWORD, path=policy)
    with pytest.raises(ValueError, match="Appoint a reviewer first"):
        enable_team_mode(WORKSPACES, path=policy, actor="root")
    assert "auto_approve" in get_default_permissions(policy), "a refused switch changes nothing"


def test_team_mode_removes_only_auto_approve_and_reports_holdouts(policy):
    add_user("rev", TEST_PASSWORD, path=policy)
    add_user("bot", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("rev", ["query", "review"], path=policy)
    set_grant("bot", "ops", ["query", "auto_approve"], path=policy)

    report = enable_team_mode(WORKSPACES, path=policy, actor="root")
    assert report.mode == "team"
    assert get_default_permissions(policy) == ["query"]
    assert report.auto_approve_holdouts == ["bot"], "explicit grants still win over defaults; the admin is told"


def test_team_mode_keeps_a_deny_by_default_policy_closed(policy):
    set_default_permissions(["auto_approve"], path=policy)
    add_user("rev", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("rev", ["query", "review"], path=policy)
    enable_team_mode(WORKSPACES, path=policy, actor="root")
    assert get_default_permissions(policy) == [], "team mode must not widen access by adding query"


def test_team_mode_requires_an_administrator(policy):
    add_user("rev", TEST_PASSWORD, path=policy)
    set_user_permissions_everywhere("rev", ["query", "review"], path=policy)
    with pytest.raises(PermissionError):
        enable_team_mode(WORKSPACES, path=policy, actor="rev")
    enable_team_mode(WORKSPACES, path=policy, actor="root")
    with pytest.raises(PermissionError):
        enable_team_mode(WORKSPACES, path=policy, actor="rev")


def test_team_mode_and_list_cli(policy, monkeypatch, capsys):
    monkeypatch.setattr(access, "ACCESS_POLICY_PATH", policy)
    monkeypatch.setattr(access, "_workspace_ids", lambda: WORKSPACES)
    monkeypatch.setattr(access, "authenticate_admin", lambda name: name)
    add_user("ann", TEST_PASSWORD, path=policy)

    assert access.main(["--as", "root", "team-mode"]) == 1
    assert "Appoint a reviewer first" in capsys.readouterr().err

    set_user_permissions_everywhere("ann", ["query", "review"], path=policy)
    assert access.main(["--as", "root", "team-mode"]) == 0
    out = capsys.readouterr().out
    assert "Review mode: team" in out
    assert "Warning: workspace 'finance'" in out, "ann is the only reviewer and cannot release her own answers"

    assert access.main(["list"]) == 0
    assert "Review mode: team" in capsys.readouterr().out


def test_status_shows_the_review_mode(rollback_index, capsys, monkeypatch):
    from workspace_manager import print_status

    monkeypatch.setattr("llm_provider.ollama_models", lambda *a, **k: None)
    print_status()
    out = capsys.readouterr().out
    assert "[OK] Review mode:" in out


def test_admin_panel_shows_the_review_mode():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "ui.py"), default_timeout=120).run()
    app.text_input[0].input("alice")
    app.text_input[1].input(TEST_PASSWORD)
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    assert any(str(c.value).startswith("Review mode:") for c in app.caption)
