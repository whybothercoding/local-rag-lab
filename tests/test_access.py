import os
import stat

import pytest

from rag_lab.security.access import (
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
from rag_lab.security.audit import list_events
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
    from rag_lab.security.access import reset_grant, set_default_permissions

    # Out-of-the-box v2 default: query + auto_approve enabled, review disabled
    add_user("erin", TEST_PASSWORD, path=policy_path)
    assert authorize("erin", "accounting", "query", path=policy_path)
    assert authorize("erin", "accounting", "auto_approve", path=policy_path)
    assert not authorize("erin", "accounting", "review", path=policy_path)

    # Admin can switch global default to strict default-deny ([])
    set_default_permissions([], path=policy_path)
    assert not authorize("erin", "accounting", "query", path=policy_path)

    set_grant("erin", "accounting", ["query"], path=policy_path)
    assert authorize("erin", "accounting", "query", path=policy_path)
    assert not authorize("erin", "accounting", "review", path=policy_path)
    assert not authorize("erin", "strategy", "query", path=policy_path)
    reset_grant("erin", "accounting", path=policy_path)
    assert not authorize("erin", "accounting", "query", path=policy_path)


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


def test_allowed_workspaces_includes_query_and_review_grants(policy_path):
    from rag_lab.security.access import set_default_permissions

    add_user("erin", TEST_PASSWORD, path=policy_path)
    # With out-of-the-box defaults (query + auto_approve), all workspaces are allowed
    assert allowed_workspaces("erin", ["accounting", "strategy", "legal"], path=policy_path) == [
        "accounting",
        "strategy",
        "legal",
    ]

    # When defaults are empty or a workspace is explicitly denied, only granted workspaces appear
    set_default_permissions([], path=policy_path)
    set_grant("erin", "accounting", ["query"], path=policy_path)
    set_grant("erin", "strategy", ["review"], path=policy_path)
    assert allowed_workspaces("erin", ["accounting", "strategy", "legal"], path=policy_path) == ["accounting", "strategy"]


def test_guardrail_denies_unauthorized_principal_before_retrieval(mocker):
    from rag_lab.pipeline.entry import guardrail_input_node

    retrieval = mocker.patch("rag_lab.pipeline.retrieval.hybrid_retrieval_node")
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
    from rag_lab.pipeline.entry import guardrail_input_node

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
    from rag_lab.pipeline.entry import guardrail_input_node

    result = guardrail_input_node({"question": "hello", "workspace_id": "default", "telemetry": {}})
    assert result["is_safe"] is False


def test_guardrail_ignores_a_bare_name_without_a_token():
    from rag_lab.pipeline.entry import guardrail_input_node

    result = guardrail_input_node({
        "question": "hello",
        "workspace_id": "default",
        "principal": "alice",
        "telemetry": {},
    })
    assert result["is_safe"] is False


def test_reviewer_without_grant_cannot_release_an_answer(mocker):
    from rag_lab.pipeline.review import human_in_the_loop_node

    mocker.patch("rag_lab.pipeline.review.interrupt", return_value={"decision": "approved", "approver_token": token_for("carol")})
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
    from rag_lab.pipeline.review import human_in_the_loop_node

    mocker.patch("rag_lab.pipeline.review.interrupt", return_value={"decision": "approved", "approver_token": token_for("bob")})
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
    from rag_lab.pipeline.review import human_in_the_loop_node

    mocker.patch("rag_lab.pipeline.review.interrupt", return_value={"decision": "approved", "approver_token": token_for("bob")})
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
    import rag_lab.ops.approve as approve
    mocker.patch("rag_lab.ops.approve.getpass.getpass", return_value=TEST_PASSWORD)
    assert verify_token(approve.confirm_reviewer(thread_state, "bob")) == "bob"
    assert "Forgot the password" in capsys.readouterr().out


def test_cli_reviewer_gets_attempts_left_and_can_retry(mocker, thread_state, capsys):
    import rag_lab.ops.approve as approve
    mocker.patch("rag_lab.ops.approve.getpass.getpass", side_effect=["wrong-password-value", TEST_PASSWORD])
    assert verify_token(approve.confirm_reviewer(thread_state, "bob")) == "bob"
    assert "Attempts left: 2" in capsys.readouterr().out


def test_cli_reviewer_is_refused_after_three_failures(mocker, thread_state):
    import rag_lab.ops.approve as approve
    prompt = mocker.patch("rag_lab.ops.approve.getpass.getpass", return_value="wrong-password-value")
    with pytest.raises(SystemExit, match="too many times"):
        approve.confirm_reviewer(thread_state, "bob")
    assert prompt.call_count == 3


def test_cli_reviewer_can_cancel_without_recording_anything(mocker, thread_state):
    import rag_lab.ops.approve as approve
    mocker.patch("rag_lab.ops.approve.getpass.getpass", side_effect=KeyboardInterrupt)
    with pytest.raises(SystemExit, match="Cancelled"):
        approve.confirm_reviewer(thread_state, "bob")


def test_cli_reviewer_needs_review_grant_even_with_correct_password(mocker, thread_state):
    import rag_lab.ops.approve as approve
    mocker.patch("rag_lab.ops.approve.getpass.getpass", return_value=TEST_PASSWORD)
    with pytest.raises(SystemExit, match="no review permission"):
        approve.confirm_reviewer(thread_state, "carol")


def test_cli_reviewer_cannot_approve_own_query(mocker):
    import rag_lab.ops.approve as approve
    from types import SimpleNamespace

    mocker.patch("rag_lab.ops.approve.getpass.getpass", return_value=TEST_PASSWORD)
    own_state = SimpleNamespace(values={"workspace_id": "default", "principal": "bob"})
    with pytest.raises(SystemExit, match="Separation of Duties"):
        approve.confirm_reviewer(own_state, "bob")


def test_password_reset_keeps_grants(policy_path):
    from rag_lab.security.access import set_password

    add_user("erin", TEST_PASSWORD, path=policy_path)
    set_grant("erin", "accounting", ["query", "review"], path=policy_path)

    new_password = "a-new-long-password"
    set_password("erin", new_password, path=policy_path)

    assert authenticate("erin", new_password, path=policy_path)
    assert not authenticate("erin", TEST_PASSWORD, path=policy_path)
    assert authorize("erin", "accounting", "review", path=policy_path)


def test_password_reset_requires_an_existing_account(policy_path):
    from rag_lab.security.access import set_password

    with pytest.raises(ValueError, match="does not exist"):
        set_password("nobody", "a-new-long-password", path=policy_path)


def test_adding_an_existing_user_points_to_password_reset(policy_path):
    add_user("erin", TEST_PASSWORD, path=policy_path)
    with pytest.raises(ValueError, match="set-password ACCOUNT=erin"):
        add_user("erin", TEST_PASSWORD, path=policy_path)


def test_delete_user_removes_account_and_grants(policy_path):
    from rag_lab.security.access import delete_user, load_policy

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
    from rag_lab.security.access import delete_user

    with pytest.raises(ValueError, match="does not exist"):
        delete_user("ghost_user", path=policy_path)


def test_reset_users_removes_all_accounts(policy_path):
    from rag_lab.security.access import load_policy, reset_users

    add_user("user_a", TEST_PASSWORD, path=policy_path)
    add_user("user_b", TEST_PASSWORD, path=policy_path)
    assert len(list_users(path=policy_path)) == 2

    reset_users(path=policy_path)

    assert list_users(path=policy_path) == []
    policy = load_policy(path=policy_path)
    assert policy["users"] == {}


def test_cli_delete_and_reset_user_commands(policy_path, monkeypatch, capsys):
    import rag_lab.security.access as access
    monkeypatch.setattr(access, "ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)
    add_user("alice", TEST_PASSWORD, admin=True, path=policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path)

    # Without --as, CLI refuses mutation
    assert access.main(["delete-user", "erin", "-y"]) == 1

    # Cancelled deletion
    monkeypatch.setattr("builtins.input", lambda _: "n")
    ret = access.main(["--as", "alice", "delete-user", "erin"])
    assert ret == 0
    assert "erin" in [u["username"] for u in list_users(path=policy_path)]

    # Confirmed deletion with -y
    ret = access.main(["--as", "alice", "delete-user", "erin", "-y"])
    assert ret == 0
    assert "erin" not in [u["username"] for u in list_users(path=policy_path)]

    # Cannot delete the last administrator
    assert access.main(["--as", "alice", "delete-user", "alice", "-y"]) == 1

    # Reset users with -y
    add_user("frank", TEST_PASSWORD, path=policy_path)
    ret = access.main(["--as", "alice", "reset-users", "-y"])
    assert ret == 0
    assert list_users(path=policy_path) == []


def test_cli_grant_and_revoke_multiple_workspaces(policy_path, monkeypatch, capsys):
    import rag_lab.security.access as access
    from rag_lab.security.access import authorize, load_policy

    monkeypatch.setattr(access, "ACCESS_POLICY_PATH", policy_path)
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", lambda _="": TEST_PASSWORD)
    monkeypatch.setattr("rag_lab.storage.workspace_manager.list_workspaces", lambda: ["accounting", "strategy", "legal"])
    add_user("alice", TEST_PASSWORD, admin=True, path=policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path)

    # Non-admin --as is rejected
    assert access.main(["--as", "erin", "grant", "erin", "accounting", "--perms", "query"]) == 1

    # 1. Comma-separated grant
    ret = access.main(["--as", "alice", "grant", "erin", "accounting,strategy", "--perms", "query,review"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert policy["users"]["erin"]["grants"]["accounting"] == ["query", "review"]
    assert policy["users"]["erin"]["grants"]["strategy"] == ["query", "review"]
    assert "legal" not in policy["users"]["erin"]["grants"]

    # 2. Grant all
    ret = access.main(["--as", "alice", "grant", "erin", "all", "--perms", "query"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert set(policy["users"]["erin"]["grants"].keys()) == {"accounting", "strategy", "legal"}
    assert policy["users"]["erin"]["grants"]["legal"] == ["query"]
    assert policy["users"]["erin"]["default_permissions"] == ["query"]

    # 3. Comma-separated revoke (sets explicit denial [] so defaults don't re-grant access)
    ret = access.main(["--as", "alice", "revoke", "erin", "accounting,strategy"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert policy["users"]["erin"]["grants"]["accounting"] == []
    assert policy["users"]["erin"]["grants"]["strategy"] == []
    assert not authorize("erin", "accounting", "query", path=policy_path)
    assert not authorize("erin", "strategy", "query", path=policy_path)
    assert authorize("erin", "legal", "query", path=policy_path)

    # 4. Revoke all
    ret = access.main(["--as", "alice", "revoke", "erin", "all"])
    assert ret == 0
    policy = load_policy(path=policy_path)
    assert policy["users"]["erin"]["grants"] == {}
    assert policy["users"]["erin"]["default_permissions"] == []
    assert not authorize("erin", "legal", "query", path=policy_path)
    assert not authorize("erin", "future_ws", "query", path=policy_path)

    # 5. Grant/revoke on nonexistent user returns error code 1 even for 'all'
    assert access.main(["--as", "alice", "grant", "ghost", "all"]) == 1
    assert access.main(["--as", "alice", "revoke", "ghost", "all"]) == 1


def test_cli_reviewer_rejects_empty_account_name(mocker, thread_state):
    import rag_lab.ops.approve as approve
    mocker.patch("builtins.input", return_value="   ")
    with pytest.raises(SystemExit, match="Account name is required"):
        approve.confirm_reviewer(thread_state, None)


def test_approve_list_inspect_and_process_decision(mocker, capsys):
    import rag_lab.ops.approve as approve
    from types import SimpleNamespace

    interrupt_obj = SimpleNamespace(
        value={
            "action": "verify_and_approve_rag_response",
            "question": "What is the rollback SLA?",
            "candidate_answer": "5 minutes for Tier-1.",
            "citations": ["runbook.md (chunk 0)"],
        }
    )
    paused_state = SimpleNamespace(
        next=("human_approval",),
        tasks=[SimpleNamespace(interrupts=[interrupt_obj])],
        values={
            "workspace_id": "default",
            "principal": "carol",
            "original_question": "What is the rollback SLA?",
            "generation": "5 minutes for Tier-1.",
            "citations": ["runbook.md (chunk 0)"],
            "degraded": ["reranker_skipped"],
        },
    )
    done_state = SimpleNamespace(
        next=(),
        tasks=[],
        values={"workspace_id": "default", "generation": "5 minutes for Tier-1."},
    )

    fake_app = mocker.MagicMock()

    def get_state_side_effect(config):
        tid = config["configurable"]["thread_id"]
        if tid == "thread-pending":
            return paused_state
        return done_state

    fake_app.get_state.side_effect = get_state_side_effect
    fake_app.stream.return_value = [{"human_approval": {"generation": "5 minutes for Tier-1."}}]
    mocker.patch("rag_lab.ops.approve.fetch_checkpoint_threads", return_value=["thread-pending", "thread-done"])

    # 1. list_pending_threads finds only paused threads with active interrupts
    pending = approve.list_pending_threads(fake_app)
    assert pending == ["thread-pending"]

    # 2. inspect_thread returns state for paused thread and None for non-paused thread
    assert approve.inspect_thread(fake_app, "thread-pending") is paused_state
    out = capsys.readouterr().out
    assert "PENDING AUDIT FOR THREAD: thread-pending" in out
    assert "reranker_skipped" in out
    assert "5 minutes for Tier-1." in out

    assert approve.inspect_thread(fake_app, "thread-done") is None
    assert "not awaiting approval" in capsys.readouterr().out

    # 3. process_decision streams Command(resume=...) and prints final state
    fake_app.get_state.side_effect = None
    fake_app.get_state.return_value = done_state
    approve.process_decision(fake_app, "thread-pending", "approved", token_for("bob"), reason="Looks accurate")
    assert fake_app.stream.called
    assert "EXECUTION RESUMED & COMPLETED" in capsys.readouterr().out


def test_approve_main_cli_modes(mocker, monkeypatch, capsys):
    import rag_lab.ops.approve as approve
    from contextlib import contextmanager
    from types import SimpleNamespace

    interrupt_obj = SimpleNamespace(
        value={
            "action": "verify_and_approve_rag_response",
            "question": "Q?",
            "candidate_answer": "A.",
            "citations": [],
        }
    )
    paused_state = SimpleNamespace(
        next=("human_approval",),
        tasks=[SimpleNamespace(interrupts=[interrupt_obj])],
        values={"workspace_id": "default", "principal": "carol", "generation": "A."},
    )
    done_state = SimpleNamespace(next=(), tasks=[], values={"workspace_id": "default", "generation": "A."})

    fake_app = mocker.MagicMock()
    fake_app.stream.return_value = [{"human_approval": {"generation": "A."}}]

    @contextmanager
    def dummy_checkpointer():
        yield mocker.MagicMock()

    mocker.patch("rag_lab.ops.approve.get_checkpointer", dummy_checkpointer)
    mocker.patch("rag_lab.ops.approve.workflow.compile", return_value=fake_app)
    mocker.patch("rag_lab.ops.approve.getpass.getpass", return_value=TEST_PASSWORD)

    # Mode 1: --list with pending thread and with empty queue
    mocker.patch("rag_lab.ops.approve.fetch_checkpoint_threads", return_value=["t-1"])
    fake_app.get_state.return_value = paused_state
    monkeypatch.setattr("sys.argv", ["approve.py", "--list"])
    approve.main()
    assert "1. t-1" in capsys.readouterr().out

    mocker.patch("rag_lab.ops.approve.fetch_checkpoint_threads", return_value=[])
    monkeypatch.setattr("sys.argv", ["approve.py", "--list"])
    approve.main()
    assert "No threads currently paused" in capsys.readouterr().out

    # Mode 2: --thread & --action on a valid pending thread
    fake_app.get_state.side_effect = [paused_state, done_state]
    monkeypatch.setattr(
        "sys.argv",
        ["approve.py", "--thread", "t-1", "--action", "approved", "--approver", "bob", "--reason", "LGTM"],
    )
    approve.main()
    assert "EXECUTION RESUMED & COMPLETED" in capsys.readouterr().out

    # Mode 2: --thread & --action on a non-pending thread exits with code 1
    fake_app.get_state.side_effect = None
    fake_app.get_state.return_value = done_state
    monkeypatch.setattr("sys.argv", ["approve.py", "--thread", "t-done", "--action", "approved", "--approver", "bob"])
    with pytest.raises(SystemExit) as excinfo:
        approve.main()
    assert excinfo.value.code == 1

    # Mode 3: Interactive auto-select when no pending threads exist exits with code 0
    mocker.patch("rag_lab.ops.approve.fetch_checkpoint_threads", return_value=[])
    monkeypatch.setattr("sys.argv", ["approve.py"])
    with pytest.raises(SystemExit) as excinfo:
        approve.main()
    assert excinfo.value.code == 0

    # Mode 3: Interactive auto-select with 'r' (reject) and 'q' (abort)
    mocker.patch("rag_lab.ops.approve.fetch_checkpoint_threads", return_value=["t-1"])
    fake_app.get_state.side_effect = [paused_state, paused_state, done_state]
    monkeypatch.setattr("sys.argv", ["approve.py", "--approver", "bob"])
    monkeypatch.setattr("builtins.input", lambda _: "r")
    approve.main()
    assert "EXECUTION RESUMED & COMPLETED" in capsys.readouterr().out

    fake_app.get_state.side_effect = [paused_state, paused_state]
    monkeypatch.setattr("builtins.input", lambda _: "q")
    approve.main()
    assert "Aborted without changes." in capsys.readouterr().out


def test_auto_approve_permission_and_grant(policy_path):
    from rag_lab.security.access import add_user, allowed_workspaces, authorize, set_grant

    # Out of the box, new users get query + auto_approve by default
    add_user("dave", TEST_PASSWORD, path=policy_path)
    assert authorize("dave", "accounting", "query", path=policy_path)
    assert authorize("dave", "accounting", "auto_approve", path=policy_path)
    assert not authorize("dave", "accounting", "review", path=policy_path)

    # Admin can disable auto_approve on a workspace (requiring human review) or deny ops
    set_grant("dave", "accounting", ["query"], path=policy_path)
    set_grant("dave", "ops", [], path=policy_path)
    assert authorize("dave", "accounting", "query", path=policy_path)
    assert not authorize("dave", "accounting", "auto_approve", path=policy_path)

    # Test granting auto_approve explicitly with hyphen or underscore
    set_grant("dave", "accounting", ["query", "auto-approve"], path=policy_path)
    assert authorize("dave", "accounting", "query", path=policy_path)
    assert authorize("dave", "accounting", "auto_approve", path=policy_path)
    assert not authorize("dave", "accounting", "review", path=policy_path)

    workspaces = allowed_workspaces("dave", ["accounting", "ops"], path=policy_path)
    assert "accounting" in workspaces
    assert "ops" not in workspaces


def test_admin_only_enforcement_and_last_admin_protection(policy_path):
    from rag_lab.security.access import (
        add_user,
        clear_user_permissions_override,
        delete_user,
        revoke_sessions,
        set_admin,
        set_default_permissions,
        set_grant,
        set_password,
        set_user_clearance,
        set_user_permissions_everywhere,
    )

    add_user("alice", TEST_PASSWORD, admin=True, path=policy_path)
    add_user("bob", TEST_PASSWORD, admin=False, path=policy_path, actor="alice")

    # Non-admin actor is refused on every administrative mutation
    with pytest.raises(PermissionError, match="not an administrator"):
        add_user("charlie", TEST_PASSWORD, path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_grant("bob", "default", ["query", "review"], path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_user_permissions_everywhere("bob", ["query", "review"], path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        clear_user_permissions_override("bob", path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_default_permissions(["query"], path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_user_clearance("bob", "secret", path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_admin("bob", True, path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        set_password("alice", "another-valid-password-123", path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        revoke_sessions("alice", path=policy_path, actor="bob")
    with pytest.raises(PermissionError, match="not an administrator"):
        delete_user("alice", path=policy_path, actor="bob")

    # A user can revoke their own sessions on sign-out
    revoke_sessions("bob", path=policy_path, actor="bob")

    # Last administrator cannot be deleted or demoted
    with pytest.raises(ValueError, match="last administrator"):
        delete_user("alice", path=policy_path, actor="alice")
    with pytest.raises(ValueError, match="last administrator"):
        set_admin("alice", False, path=policy_path, actor="alice")


def test_token_revocation_on_password_change_delete_and_recreate(policy_path):
    from rag_lab.security.access import add_user, delete_user, issue_token, revoke_sessions, set_password, verify_token

    add_user("alice", TEST_PASSWORD, admin=True, path=policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path, actor="alice")
    tok1 = issue_token("erin", path=policy_path)
    assert verify_token(tok1, path=policy_path) == "erin"

    # 1. Explicit session revocation invalidates existing token
    revoke_sessions("erin", path=policy_path, actor="alice")
    assert verify_token(tok1, path=policy_path) == ""

    # 2. Password change invalidates existing token
    tok2 = issue_token("erin", path=policy_path)
    assert verify_token(tok2, path=policy_path) == "erin"
    set_password("erin", "new-strong-password-999", path=policy_path, actor="alice")
    assert verify_token(tok2, path=policy_path) == ""

    # 3. Deleting and recreating an account with the same name does not resurrect old tokens
    tok3 = issue_token("erin", path=policy_path)
    assert verify_token(tok3, path=policy_path) == "erin"
    delete_user("erin", path=policy_path, actor="alice")
    assert verify_token(tok3, path=policy_path) == ""
    add_user("erin", TEST_PASSWORD, path=policy_path, actor="alice")
    assert verify_token(tok3, path=policy_path) == ""


def test_concurrent_policy_updates_do_not_drop_writes(policy_path):
    import threading
    from rag_lab.security.access import add_user, load_policy, set_grant

    add_user("alice", TEST_PASSWORD, admin=True, path=policy_path)
    add_user("erin", TEST_PASSWORD, path=policy_path, actor="alice")

    def grant_worker(idx: int) -> None:
        set_grant("erin", f"ws_{idx}", ["query"], path=policy_path, actor="alice")

    threads = [threading.Thread(target=grant_worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    grants = load_policy(policy_path)["users"]["erin"]["grants"]
    assert {f"ws_{i}" for i in range(12)} <= set(grants.keys())


def test_reviewer_clearance_hierarchy_and_auto_approve_safety_gates(mocker):
    from rag_lab.security.access import set_grant, set_user_clearance
    from rag_lab.pipeline.state import ABSTAIN_MESSAGE
    from rag_lab.pipeline.review import human_in_the_loop_node

    # 1. Auto-approve user STILL pauses at human review when answer failed grading or has security warning
    set_grant("carol", "default", ["query", "auto_approve"])
    try:
        mocker.patch("rag_lab.pipeline.review.interrupt", return_value={"decision": "approved", "approver_token": token_for("alice")})
        res_ungrounded = human_in_the_loop_node({
            "workspace_id": "default",
            "principal": "carol",
            "principal_clearance": "public",
            "sanitized_question": "what is the sla",
            "generation": "Unverified claim after circuit break.",
            "citations": ["[c1] doc.md"],
            "hallucination_status": "circuit_break",
            "relevance_status": "relevant",
            "degraded": [],
            "history": [],
            "telemetry": {},
        })
        assert res_ungrounded["generation"] == "Unverified claim after circuit break."
        event = list_events(limit=1, workspace_id="default")[0]
        assert event["data"]["approver"] == "alice"
        assert "groundedness check did not pass (circuit_break)" in event["data"]["review_reasons"]

        # 2. Abstained answers are never stored in semantic cache even when approved
        from rag_lab.retrieval.semantic_cache import lookup_cache
        human_in_the_loop_node({
            "workspace_id": "default",
            "principal": "carol",
            "principal_clearance": "public",
            "original_question": "Unanswerable question?",
            "sanitized_question": "unanswerable question?",
            "generation": ABSTAIN_MESSAGE,
            "citations": [],
            "hallucination_status": "grounded",
            "relevance_status": "relevant",
            "degraded": [],
            "history": [],
            "telemetry": {"abstained": True},
        })
        assert lookup_cache("default", "Unanswerable question?", "public")["hit"] is False
    finally:
        set_grant("carol", "default", ["query"])

    # 3. Reviewer with lower clearance ('public') cannot approve a 'secret' requester's query
    set_user_clearance("alice", "secret")
    set_user_clearance("bob", "public")
    try:
        mocker.patch("rag_lab.pipeline.review.interrupt", return_value={"decision": "approved", "approver_token": token_for("bob")})
        res_clr = human_in_the_loop_node({
            "workspace_id": "default",
            "principal": "alice",
            "principal_clearance": "secret",
            "sanitized_question": "secret question",
            "generation": "Secret answer.",
            "citations": ["[c1] secret.md"],
            "hallucination_status": "grounded",
            "relevance_status": "relevant",
            "degraded": [],
            "history": [],
            "telemetry": {},
        })
        assert "rejected by human compliance" in res_clr["generation"].lower()
        ev = list_events(limit=1, workspace_id="default")[0]
        assert ev["data"]["released"] is False
        assert "approver clearance is below the requester's" in ev["data"]["reason"]
    finally:
        set_user_clearance("alice", "public")



