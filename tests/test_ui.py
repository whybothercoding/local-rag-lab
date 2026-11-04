"""Streamlit UI smoke tests: every page an account can reach renders without errors, and admin panels work."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_lab.security.access import add_user, authorize, delete_user, set_grant, set_user_permissions_everywhere
from conftest import TEST_PASSWORD

UI = str(Path(__file__).resolve().parent.parent / "ui.py")
ADMIN_PANELS = {
    "Audit log (administrators)",
    "User Grants & Auto-Approval (Admin)",
    "Add User Account (Admin)",
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
