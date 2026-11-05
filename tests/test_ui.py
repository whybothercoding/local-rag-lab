"""Streamlit UI smoke tests: every page an account can reach renders without errors, and admin panels work."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_lab.security.access import (
    add_user,
    authorize,
    delete_group,
    delete_user,
    list_groups,
    load_policy,
    set_grant,
    set_user_permissions_everywhere,
)
from conftest import TEST_PASSWORD

UI = str(Path(__file__).resolve().parent.parent / "ui.py")
ADMIN_PANELS = {
    "Audit log (administrators)",
    "User Grants & Auto-Approval (Admin)",
    "Add User Account (Admin)",
    "Groups (Admin)",
    "Two-factor sign-in (Admin)",
}


def _signed_in(name: str) -> AppTest:
    app = AppTest.from_file(UI, default_timeout=120).run()
    app.text_input[0].input(name)
    app.text_input[1].input(TEST_PASSWORD)
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    assert not app.exception, app.exception
    return app


@pytest.fixture
def grant_target():
    """A throwaway account for the grants panel; removed afterwards so the shared policy stays as seeded."""
    add_user("uigrant", TEST_PASSWORD)
    set_user_permissions_everywhere("uigrant", [])
    set_grant("uigrant", "accounting", ["query"])
    yield "uigrant"
    delete_user("uigrant")


def test_admin_page_renders_every_panel_without_errors():
    app = _signed_in("alice")
    labels = {e.label for e in app.sidebar.expander}
    assert ADMIN_PANELS <= labels, labels


def test_non_admin_page_has_no_admin_panels():
    app = _signed_in("carol")
    labels = {e.label for e in app.sidebar.expander}
    assert not (ADMIN_PANELS & labels), labels


def test_grants_panel_shows_and_saves_effective_permissions(grant_target):
    app = _signed_in("alice")
    app.selectbox(key="grant_user_select").set_value(grant_target).run()
    app.selectbox(key="grant_ws_select").set_value("accounting").run()
    assert not app.exception, app.exception

    prefix = f"{grant_target}_accounting"
    assert app.checkbox(key=f"g_q_{prefix}").value is True
    assert app.checkbox(key=f"g_r_{prefix}").value is False

    app.checkbox(key=f"g_r_{prefix}").check().run()
    app.button(key=f"save_grants_{prefix}").click().run()
    assert not app.exception, app.exception
    assert authorize(grant_target, "accounting", "review")

    # The "all" view reads the account's permissions on the active workspace and must render too.
    app.selectbox(key="grant_ws_select").set_value("all").run()
    assert not app.exception, app.exception


def test_role_select_changes_an_accounts_role(grant_target):
    app = _signed_in("alice")
    app.selectbox(key="grant_user_select").set_value(grant_target).run()
    app.selectbox(key=f"role_select_{grant_target}").set_value("reviewer").run()
    app.button(key=f"save_role_{grant_target}").click().run()
    assert not app.exception, app.exception
    assert load_policy()["users"][grant_target]["role"] == "reviewer"


def test_groups_panel_creates_a_group_adds_a_member_and_grants_a_role(grant_target):
    app = _signed_in("alice")
    try:
        app.text_input(key="new_group_name").input("uiteam").run()
        app.button(key="create_group_btn").click().run()
        assert not app.exception, app.exception
        app.selectbox(key="group_select").set_value("uiteam").run()
        app.selectbox(key="group_member_select").set_value(grant_target).run()
        app.button(key="group_member_btn").click().run()
        app.selectbox(key="group_ws_select").set_value("all").run()
        app.selectbox(key="group_role_select").set_value("member").run()
        app.button(key="group_grant_btn").click().run()
        assert not app.exception, app.exception
        assert list_groups() == [{"group": "uiteam", "members": [grant_target], "grants": {"*": "member"}}]
        assert authorize(grant_target, "accounting", "query"), "the account's own grant still wins"
    finally:
        delete_group("uiteam")


def test_add_user_panel_warns_when_answers_could_not_be_reviewed():
    app = _signed_in("alice")
    try:
        app.text_input(key="new_user_name").input("uinewbie").run()
        app.text_input(key="new_user_pass").input(TEST_PASSWORD).run()
        app.selectbox(key="new_user_role").set_value("member").run()
        app.button(key="create_user_btn").click().run()
        assert not app.exception, app.exception
        assert load_policy()["users"]["uinewbie"]["role"] == "member"
        assert any("Created account 'uinewbie'" in s.value for s in app.sidebar.success)
        assert any("nobody able to approve" in w.value for w in app.sidebar.warning)
    finally:
        delete_user("uinewbie")


def test_every_screen_shows_the_one_rejection_notice():
    """The reviewer's session, a reloaded thread and the requester's screen must read the same rejection text."""
    source = (Path(UI).parent / "rag_lab" / "ui" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "REJECTION_NOTICE" in source
    assert "rejected by" not in source.lower(), "a rejection text is written inline instead of using REJECTION_NOTICE"


def test_session_restores_on_reload_via_query_params():
    """When a signed session token is present in query params, reload bypasses login and restores user state."""
    set_grant("alice", "strategy", ["query", "review"])
    try:
        app = _signed_in("alice")
        token = app.query_params.get("session")
        assert token, "login must record session token in query params"

        # Simulate browser reload with clean session state but preserved URL query params
        reloaded = AppTest.from_file(UI, default_timeout=120)
        reloaded.query_params["session"] = token
        reloaded.query_params["ws"] = "strategy"
        reloaded.run()
        assert not reloaded.exception, reloaded.exception
        assert reloaded.session_state.get("user") == "alice"
        assert reloaded.sidebar.selectbox[0].value == "strategy"
    finally:
        set_grant("alice", "strategy", [])


def test_sign_out_clears_session_and_query_params():
    """Sign out revokes session and clears query params so reload stays on login."""
    app = _signed_in("alice")
    assert app.query_params.get("session")
    sign_out_btn = [b for b in app.sidebar.button if b.label == "Sign out"][0]
    sign_out_btn.click().run()
    assert not app.exception, app.exception
    assert not app.session_state.get("user")
    assert not app.query_params.get("session")
