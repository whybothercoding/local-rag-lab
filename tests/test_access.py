import os
import stat

import pytest

from access import (
    add_user,
    allowed_workspaces,
    authenticate,
    authorize,
    hash_password,
    list_users,
    set_grant,
    verify_password,
    verify_token,
)
from audit import list_events
from conftest import TEST_PASSWORD, token_for


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
        "principal_token": token_for("dave"),
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
        "principal_token": token_for("carol"),
        "telemetry": {},
    })
    assert result["is_safe"] is True
    assert result["principal"] == "carol"


def test_guardrail_denies_anonymous_caller():
    from app import guardrail_input_node

    result = guardrail_input_node({"question": "hello", "workspace_id": "default", "telemetry": {}})
    assert result["is_safe"] is False


def test_guardrail_ignores_a_bare_name_without_a_token():
    from app import guardrail_input_node

    result = guardrail_input_node({
        "question": "hello",
        "workspace_id": "default",
        "principal": "alice",
        "telemetry": {},
    })
    assert result["is_safe"] is False


def test_reviewer_without_grant_cannot_release_an_answer(mocker):
    from app import human_in_the_loop_node

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver_token": token_for("carol")})
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

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver_token": token_for("bob")})
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

    mocker.patch("app.interrupt", return_value={"decision": "approved", "approver_token": token_for("bob")})
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


@pytest.fixture
def thread_state():
    from types import SimpleNamespace

    return SimpleNamespace(values={"workspace_id": "default"})


def test_cli_reviewer_signs_in_with_correct_password(mocker, thread_state, capsys):
    import approve

    mocker.patch("approve.getpass.getpass", return_value=TEST_PASSWORD)
    assert verify_token(approve.confirm_reviewer(thread_state, "bob")) == "bob"
    assert "Forgot the password" in capsys.readouterr().out


def test_cli_reviewer_gets_attempts_left_and_can_retry(mocker, thread_state, capsys):
    import approve

    mocker.patch("approve.getpass.getpass", side_effect=["wrong-password-value", TEST_PASSWORD])
    assert verify_token(approve.confirm_reviewer(thread_state, "bob")) == "bob"
    assert "Attempts left: 2" in capsys.readouterr().out


def test_cli_reviewer_is_refused_after_three_failures(mocker, thread_state):
    import approve

    prompt = mocker.patch("approve.getpass.getpass", return_value="wrong-password-value")
    with pytest.raises(SystemExit, match="too many times"):
        approve.confirm_reviewer(thread_state, "bob")
    assert prompt.call_count == 3


def test_cli_reviewer_can_cancel_without_recording_anything(mocker, thread_state):
    import approve

    mocker.patch("approve.getpass.getpass", side_effect=KeyboardInterrupt)
    with pytest.raises(SystemExit, match="Cancelled"):
        approve.confirm_reviewer(thread_state, "bob")


def test_cli_reviewer_needs_review_grant_even_with_correct_password(mocker, thread_state):
    import approve

    mocker.patch("approve.getpass.getpass", return_value=TEST_PASSWORD)
    with pytest.raises(SystemExit, match="no review permission"):
        approve.confirm_reviewer(thread_state, "carol")


def test_password_reset_keeps_grants(policy_path):
    from access import set_password

    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query", "review"], path=policy_path)

    new_password = "a-new-long-password"
    set_password("erin", new_password, path=policy_path)

    assert authenticate("erin", new_password, path=policy_path)
    assert not authenticate("erin", TEST_PASSWORD, path=policy_path)
    assert authorize("erin", "accounting", "review", path=policy_path)


def test_password_reset_requires_an_existing_account(policy_path):
    from access import set_password

    with pytest.raises(ValueError, match="does not exist"):
        set_password("nobody", "a-new-long-password", path=policy_path)


def test_adding_an_existing_user_points_to_password_reset(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    with pytest.raises(ValueError, match="set-password ACCOUNT=erin"):
        add_user("erin", TEST_PASSWORD, path=policy_path)


def test_delete_user_removes_account_and_grants(policy_path):
    from access import delete_user, load_policy

    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query", "review"], path=policy_path)

    delete_user("erin", path=policy_path)

    assert not authenticate("erin", TEST_PASSWORD, path=policy_path)
    assert not authorize("erin", "accounting", "query", path=policy_path)
    assert not authorize("erin", "accounting", "review", path=policy_path)
    assert "erin" not in [u["username"] for u in list_users(path=policy_path)]

    # Confirm policy HMAC integrity is intact
    policy = load_policy(path=policy_path)
    assert "erin" not in policy["users"]


def test_delete_nonexistent_user_raises(policy_path):
    from access import delete_user

    with pytest.raises(ValueError, match="does not exist"):
        delete_user("ghost_user", path=policy_path)


def test_reset_users_removes_all_accounts(policy_path):
    from access import delete_user, load_policy, reset_users

    add_user("user_a", TEST_PASSWORD, path=policy_path)
    add_user("user_b", TEST_PASSWORD, path=policy_path)
    assert len(list_users(path=policy_path)) == 2

    reset_users(path=policy_path)

    assert list_users(path=policy_path) == []
    policy = load_policy(path=policy_path)
    assert policy["users"] == {}


def test_cli_delete_and_reset_user_commands(policy_path, monkeypatch, capsys):
    import access

    monkeypatch.setattr(access, "ACCESS_POLICY_PATH", policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path)

    # Cancelled deletion
    monkeypatch.setattr("builtins.input", lambda _: "n")
    ret = access.main(["delete-user", "erin"])
    assert ret == 0
    assert "erin" in [u["username"] for u in list_users(path=policy_path)]

    # Confirmed deletion with -y
    ret = access.main(["delete-user", "erin", "-y"])
    assert ret == 0
    assert "erin" not in [u["username"] for u in list_users(path=policy_path)]

    # Reset users with -y
    add_user("frank", TEST_PASSWORD, path=policy_path)
    ret = access.main(["reset-users", "-y"])
    assert ret == 0
    assert list_users(path=policy_path) == []


def test_cli_grant_and_revoke_multiple_workspaces(policy_path, monkeypatch, capsys):
    import access
    from access import load_policy

    monkeypatch.setattr(access, "ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("workspace_manager.list_workspaces", lambda: ["accounting", "strategy", "legal"])
    add_user("erin", TEST_PASSWORD, path=policy_path)

    # 1. Comma-separated grant
    ret = access.main(["grant", "erin", "accounting,strategy", "--perms", "query,review"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert policy["users"]["erin"]["grants"]["accounting"] == ["query", "review"]
    assert policy["users"]["erin"]["grants"]["strategy"] == ["query", "review"]
    assert "legal" not in policy["users"]["erin"]["grants"]

    # 2. Grant all
    ret = access.main(["grant", "erin", "all", "--perms", "query"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert set(policy["users"]["erin"]["grants"].keys()) == {"accounting", "strategy", "legal"}
    assert policy["users"]["erin"]["grants"]["legal"] == ["query"]

    # 3. Comma-separated revoke
    ret = access.main(["revoke", "erin", "accounting,strategy"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert "accounting" not in policy["users"]["erin"]["grants"]
    assert "strategy" not in policy["users"]["erin"]["grants"]
    assert "legal" in policy["users"]["erin"]["grants"]

    # 4. Revoke all
    ret = access.main(["revoke", "erin", "all"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert policy["users"]["erin"]["grants"] == {}

