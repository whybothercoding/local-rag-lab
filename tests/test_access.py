import os
import stat

import pytest

import access
from access import (
    add_user,
    allowed_workspaces,
    authenticate,
    authorize,
    hash_password,
    list_users,
    set_grant,
    verify_password,
)
from audit import list_events
from conftest import TEST_PASSWORD


@pytest.fixture
def policy_path(tmp_path):
    return str(tmp_path / "access_policy.json")


def test_password_hash_round_trip():
    stored = hash_password(TEST_PASSWORD)
    assert stored.startswith("scrypt$")
    assert TEST_PASSWORD not in stored
    assert verify_password(TEST_PASSWORD, stored)
    assert not verify_password("wrong-password-entirely", stored)


def test_malformed_stored_hash_never_verifies():
    assert verify_password("anything", "") is False
    assert verify_password("anything", "plaintext") is False
    assert verify_password("anything", "md5$1$2$3$aa$bb") is False


def test_add_user_validates_input(policy_path):
    with pytest.raises(ValueError, match="at least"):
        add_user("erin", "short", path=policy_path)
    with pytest.raises(ValueError, match="Usernames"):
        add_user("Erin Smith!", TEST_PASSWORD, path=policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path)
    with pytest.raises(ValueError, match="already exists"):
        add_user("erin", TEST_PASSWORD, path=policy_path)


def test_authentication_accepts_only_the_right_password(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    assert authenticate("erin", TEST_PASSWORD, path=policy_path)
    assert authenticate("ERIN", TEST_PASSWORD, path=policy_path)
    assert not authenticate("erin", "not-the-password-at-all", path=policy_path)
    assert not authenticate("nobody", TEST_PASSWORD, path=policy_path)


def test_authorization_is_default_deny(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    assert not authorize("erin", "accounting", "query", path=policy_path)

    set_grant("erin", "accounting", ["query"], path=policy_path)
    assert authorize("erin", "accounting", "query", path=policy_path)
    assert not authorize("erin", "accounting", "review", path=policy_path)
    assert not authorize("erin", "strategy", "query", path=policy_path)


def test_revoke_removes_access(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query", "review"], path=policy_path)
    set_grant("erin", "accounting", [], path=policy_path)
    assert not authorize("erin", "accounting", "query", path=policy_path)


def test_missing_policy_file_denies_everything(tmp_path):
    missing = str(tmp_path / "does-not-exist.json")
    assert not authorize("erin", "accounting", "query", path=missing)
    assert list_users(path=missing) == []


def test_unknown_permission_is_rejected(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    with pytest.raises(ValueError):
        authorize("erin", "accounting", "delete", path=policy_path)
    with pytest.raises(ValueError, match="Unknown permission"):
        set_grant("erin", "accounting", ["delete"], path=policy_path)


def test_policy_file_is_owner_only_and_hashes_are_not_listed(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    assert stat.S_IMODE(os.stat(policy_path).st_mode) == 0o600
    listing = str(list_users(path=policy_path))
    assert "scrypt$" not in listing and "password" not in listing


def test_allowed_workspaces_filters_by_query_grant(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query"], path=policy_path)
    set_grant("erin", "strategy", ["review"], path=policy_path)
    assert allowed_workspaces("erin", ["accounting", "strategy", "legal"], path=policy_path) == ["accounting"]


def test_guardrail_denies_unauthorized_principal_before_retrieval(mocker):
    from app import guardrail_input_node

    retrieval = mocker.patch("app.hybrid_retrieval_node")
    result = guardrail_input_node({
        "question": "What is the Q4 roadmap?",
        "workspace_id": "default",
        "session_id": "s-denied",
        "principal": "dave",
        "telemetry": {},
    })

    assert result["is_safe"] is False
    assert "ACCESS DENIED" in result["generation"]
    retrieval.assert_not_called()
    event = list_events(limit=1, workspace_id="default")[0]
    assert event["event_type"] == "access_denied"
    assert event["data"]["principal"] == "dave"


def test_guardrail_allows_principal_with_query_grant():
    from app import guardrail_input_node

    result = guardrail_input_node({
        "question": "What is the Q4 roadmap?",
        "workspace_id": "default",
        "session_id": "s-allowed",
        "principal": "carol",
        "telemetry": {},
    })
    assert result["is_safe"] is True


def test_guardrail_denies_anonymous_caller():
    from app import guardrail_input_node

    result = guardrail_input_node({"question": "hello", "workspace_id": "default", "telemetry": {}})
    assert result["is_safe"] is False


def test_cli_reviewer_must_authenticate_and_hold_review_grant(mocker):
    from types import SimpleNamespace

    import approve

    state = SimpleNamespace(values={"workspace_id": "default"})

    mocker.patch("approve.getpass.getpass", return_value=TEST_PASSWORD)
    assert approve.confirm_reviewer(state, "bob") == "bob"

    mocker.patch("approve.getpass.getpass", return_value="wrong-password-value")
    with pytest.raises(SystemExit, match="Authentication failed"):
        approve.confirm_reviewer(state, "bob")

    # carol authenticates correctly but has query-only access, so she cannot decide.
    mocker.patch("approve.getpass.getpass", return_value=TEST_PASSWORD)
    with pytest.raises(SystemExit, match="no review permission"):
        approve.confirm_reviewer(state, "carol")


def test_reviewer_without_grant_cannot_release_an_answer(mocker):
    from app import human_in_the_loop_node

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver": "carol"})
    result = human_in_the_loop_node({
        "workspace_id": "default",
        "principal": "dave",
        "sanitized_question": "what is the roadmap",
        "generation": "EU and North American SME logistics.",
        "citations": [],
        "history": [],
        "telemetry": {},
    })
    assert "rejected by human compliance" in result["generation"].lower()
    event = list_events(limit=1, workspace_id="default")[0]
    assert event["data"]["released"] is False
    assert "lacks review permission" in event["data"]["reason"]


def test_requester_cannot_approve_their_own_query(mocker):
    from app import human_in_the_loop_node

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver": "bob"})
    result = human_in_the_loop_node({
        "workspace_id": "default",
        "principal": "bob",
        "sanitized_question": "what is the roadmap",
        "generation": "EU and North American SME logistics.",
        "citations": [],
        "history": [],
        "telemetry": {},
    })
    assert "rejected by human compliance" in result["generation"].lower()
    event = list_events(limit=1, workspace_id="default")[0]
    assert "self-approval" in event["data"]["reason"]


def test_reviewer_can_approve_another_users_query(mocker):
    from app import human_in_the_loop_node

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver": "bob"})
    result = human_in_the_loop_node({
        "workspace_id": "default",
        "principal": "carol",
        "sanitized_question": "what is the roadmap",
        "generation": "EU and North American SME logistics.",
        "citations": [],
        "history": [],
        "telemetry": {},
    })
    assert result["generation"] == "EU and North American SME logistics."
    event = list_events(limit=1, workspace_id="default")[0]
    assert event["data"]["released"] is True
    assert event["data"]["approver"] == "bob"
