"""Two-factor sign-in (TOTP): enrolment, enforcement, replay, recovery codes, resets, and every sign-in path."""
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyotp
import pytest

import rag_lab.security.access as access
from rag_lab.config import settings
from rag_lab.security.access import (
    PolicyIntegrityError,
    add_user,
    begin_totp_enrollment,
    confirm_totp_enrollment,
    disable_totp,
    load_policy,
    recovery_codes_left,
    regenerate_recovery_codes,
    set_role,
    sign_in,
    totp_enabled,
    verify_token,
)
from rag_lab.security.audit import count_events_since, list_events
from conftest import TEST_PASSWORD

T0 = 1_800_000_030.0  # a fixed instant; each 30-second step is one code


@pytest.fixture
def clock(monkeypatch):
    state = {"t": T0}
    monkeypatch.setattr(access, "_now", lambda: state["t"])
    return state


@pytest.fixture
def env(tmp_path):
    return {"path": str(tmp_path / "access_policy.json"), "audit": str(tmp_path / "audit.db")}


@pytest.fixture
def policy(env):
    add_user("root", TEST_PASSWORD, admin=True, path=env["path"])
    add_user("ann", TEST_PASSWORD, path=env["path"], role="member")
    return env["path"]


def code(secret: str, t: float, offset: int = 0) -> str:
    return pyotp.TOTP(secret).generate_otp(int(t // 30) + offset)


def enrol(policy_path: str, name: str, clock_state: dict) -> tuple[str, list[str]]:
    enrolment = begin_totp_enrollment(name, path=policy_path, actor=name)
    codes = confirm_totp_enrollment(name, code(enrolment.secret, clock_state["t"]), path=policy_path, actor=name)
    clock_state["t"] += 30  # the confirming code is spent; sign-ins use the next step
    return enrolment.secret, codes


def sign(env, name="root", pw=TEST_PASSWORD, **kw):
    return sign_in(name, pw, env["path"], env["audit"], **kw)


# --- Enrolment ------------------------------------------------------------------------------------------

def test_enrolment_is_two_step_and_stores_nothing_in_the_clear(policy, clock):
    enrolment = begin_totp_enrollment("root", path=policy, actor="root")
    assert enrolment.uri.startswith("otpauth://totp/") and "issuer=Local%20RAG%20Lab" in enrolment.uri
    assert not totp_enabled("root", policy), "a pending secret protects nothing yet"
    assert sign_in("root", TEST_PASSWORD, policy).ok, "and cannot lock the account out"

    with pytest.raises(ValueError, match="not valid"):
        confirm_totp_enrollment("root", "000000", path=policy, actor="root")
    assert not totp_enabled("root", policy)

    codes = confirm_totp_enrollment("root", code(enrolment.secret, clock["t"]), path=policy, actor="root")
    assert totp_enabled("root", policy) and len(codes) == 10 and len(set(codes)) == 10
    raw = Path(policy).read_text()
    assert enrolment.secret not in raw, "the secret is sealed"
    assert not any(c in raw for c in codes), "recovery codes are stored as keyed tags"
    assert recovery_codes_left("root", policy) == 10


def test_only_the_account_itself_enrols_and_only_owners_and_admins(policy, clock):
    with pytest.raises(PermissionError, match="the account itself"):
        begin_totp_enrollment("root", path=policy, actor="ann")
    with pytest.raises(PermissionError, match="the account itself"):
        begin_totp_enrollment("ann", path=policy, actor="root")
    with pytest.raises(PermissionError, match="the account itself"):
        begin_totp_enrollment("root", path=policy, actor=None)
    with pytest.raises(PermissionError, match="owners and administrators"):
        begin_totp_enrollment("ann", path=policy, actor="ann")
    enrol(policy, "root", clock)
    with pytest.raises(ValueError, match="already on"):
        begin_totp_enrollment("root", path=policy, actor="root")


def test_a_secret_sealed_for_one_account_does_not_open_for_another(policy, env, clock):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    secret, _ = enrol(policy, "root", clock)
    stolen = load_policy(policy)
    stolen["users"]["bea"]["totp"] = dict(stolen["users"]["root"]["totp"])  # re-signed by someone with the key
    access.save_policy(stolen, policy)
    with pytest.raises(PolicyIntegrityError, match="cannot be opened"):
        sign(env, "bea", code=code(secret, clock["t"]))


# --- Sign-in --------------------------------------------------------------------------------------------

def test_the_code_is_asked_for_only_after_the_password_checks_out(policy, env, clock):
    secret, _ = enrol(policy, "root", clock)
    wrong_pw = sign(env, pw="not-the-password-1")
    assert wrong_pw.reason == "invalid", "a wrong password never reveals that two-factor is on"

    missing = sign(env)
    assert not missing.ok and missing.reason == "totp_required" and "authenticator" in missing.describe()
    assert count_events_since("sign_in_failed", "root", datetime.now(UTC) - timedelta(minutes=5), env["audit"]) == 1, \
        "only the wrong password counted"

    ok = sign(env, code=code(secret, clock["t"]))
    assert ok.ok and verify_token(ok.token, path=policy) == "root"
    assert sign(env, "ann").ok, "accounts without two-factor are unaffected"


def test_a_wrong_code_fails_counts_toward_the_lockout_and_is_audited(policy, env, clock):
    secret, _ = enrol(policy, "root", clock)
    for _ in range(access.LOCKOUT_MAX_FAILURES):
        result = sign(env, code="123456")
        assert result.reason == "invalid_code" and "code" in result.describe()
    locked = sign(env, code=code(secret, clock["t"]))
    assert not locked.ok and locked.reason == "locked", "even the right code is refused during a lockout"
    stages = [e["data"].get("stage") for e in list_events(path=env["audit"], event_type="sign_in_failed")]
    assert stages.count("second_factor") == access.LOCKOUT_MAX_FAILURES


def test_the_prompt_callback_supplies_the_code_for_command_line_sign_ins(policy, env, clock):
    secret, _ = enrol(policy, "root", clock)
    asked = []
    result = sign(env, code_prompt=lambda: asked.append(1) or code(secret, clock["t"]))
    assert result.ok and asked == [1]
    assert sign(env, "ann", code_prompt=lambda: asked.append(2) or "").ok and asked == [1], "never asked without two-factor"


def test_a_code_works_once_and_older_steps_never_work_again(policy, env, clock):
    secret, _ = enrol(policy, "root", clock)
    now_code = code(secret, clock["t"])
    assert sign(env, code=now_code).ok
    replay = sign(env, code=now_code)
    assert replay.reason == "invalid_code", "the same code cannot be replayed"

    earlier = code(secret, clock["t"], -1)
    assert sign(env, code=earlier).reason == "invalid_code", "an older step than the last accepted is refused"

    clock["t"] += 30
    assert sign(env, code=code(secret, clock["t"], 1)).ok, "a code one step ahead is inside the drift window"
    assert sign(env, code=code(secret, clock["t"], 2)).reason == "invalid_code", "two steps ahead is outside it"


def test_two_simultaneous_sign_ins_cannot_both_spend_one_code(policy, env, clock):
    secret, _ = enrol(policy, "root", clock)
    shared = code(secret, clock["t"])
    results = []

    def attempt():
        results.append(sign(env, code=shared))

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r.ok for r in results) == 1, [r.reason for r in results]


def test_recovery_codes_work_once_and_are_counted_and_audited(policy, env, clock):
    _, codes = enrol(policy, "root", clock)
    first = sign(env, code=codes[0].upper())
    assert first.ok, "case and dashes are forgiven"
    assert sign(env, code=codes[0]).reason == "invalid_code", "a recovery code is spent when used"
    assert recovery_codes_left("root", policy) == 9
    spent = list_events(path=env["audit"], event_type="recovery_code_used")
    assert spent and spent[0]["data"]["remaining"] == 9
    assert sign(env, code=codes[0].replace("-", "")[:-1] + "0").reason == "invalid_code"


def test_regenerating_recovery_codes_retires_the_old_ones(policy, env, clock):
    secret, old = enrol(policy, "root", clock)
    with pytest.raises(ValueError, match="not valid"):
        regenerate_recovery_codes("root", "000000", path=policy, actor="root")
    new = regenerate_recovery_codes("root", code(secret, clock["t"]), path=policy, actor="root")
    assert set(new).isdisjoint(old) and recovery_codes_left("root", policy) == 10
    assert sign(env, code=old[0]).reason == "invalid_code"
    assert sign(env, code=new[0]).ok
    with pytest.raises(PermissionError):
        regenerate_recovery_codes("root", "x", path=policy, actor="ann")


def test_a_demoted_account_keeps_its_second_factor(policy, env, clock):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    secret, _ = enrol(policy, "bea", clock)
    set_role("bea", "member", path=policy, actor="root")
    assert sign(env, "bea").reason == "totp_required", "enforcement follows the enrolment, not the role"
    assert sign(env, "bea", code=code(secret, clock["t"])).ok


# --- Turning it off -------------------------------------------------------------------------------------

def test_turning_it_off_needs_a_valid_code(policy, clock):
    secret, codes = enrol(policy, "root", clock)
    with pytest.raises(ValueError, match="valid"):
        disable_totp("root", "000000", path=policy, actor="root")
    with pytest.raises(ValueError, match="valid"):
        disable_totp("root", None, path=policy, actor="root")
    assert totp_enabled("root", policy)
    disable_totp("root", codes[0], path=policy, actor="root")
    assert not totp_enabled("root", policy)
    assert "totp" not in load_policy(policy)["users"]["root"]


def test_only_an_owner_resets_someone_elses_second_factor_and_it_signs_them_out(policy, env, clock):
    add_user("bea", TEST_PASSWORD, admin=True, path=policy, actor="root")
    add_user("cy", TEST_PASSWORD, admin=True, path=policy, actor="root")
    bea_secret, _ = enrol(policy, "bea", clock)
    enrol(policy, "root", clock)
    token = sign(env, "bea", code=code(bea_secret, clock["t"])).token
    assert verify_token(token, path=policy) == "bea"

    with pytest.raises(PermissionError, match="Only an owner"):
        disable_totp("bea", None, path=policy, actor="cy")  # an admin, not an owner
    with pytest.raises(PermissionError, match="Only an owner"):
        disable_totp("root", None, path=policy, actor="bea")
    with pytest.raises(PermissionError, match="signed-in"):
        disable_totp("bea", None, path=policy, actor=None)

    disable_totp("bea", None, path=policy, actor="root")
    assert not totp_enabled("bea", policy)
    assert verify_token(token, path=policy) == "", "the reset signs the account out everywhere"
    with pytest.raises(ValueError, match="not on"):
        disable_totp("bea", None, path=policy, actor="root")


def test_changes_to_a_second_factor_are_audited(policy, env, clock):
    secret, codes = enrol(policy, "root", clock)
    disable_totp("root", codes[0], path=policy, actor="root")
    kinds = [e["event_type"] for e in list_events(path=None, limit=50)]
    assert {"totp_enrolment_started", "totp_enrolled", "totp_disabled"} <= set(kinds)
    assert secret not in json.dumps(list_events(path=None, limit=50))


# --- Command line ---------------------------------------------------------------------------------------

def test_cli_enrols_then_requires_the_code_for_every_admin_command(policy, monkeypatch, capsys, clock):
    monkeypatch.setattr(settings, "access_policy_path", policy)
    prompts = []

    def getpass(prompt=""):
        prompts.append(prompt)
        return TEST_PASSWORD if prompt.startswith("Password") else code(secrets["root"], clock["t"])

    secrets: dict[str, str] = {}
    monkeypatch.setattr("rag_lab.security.access.getpass.getpass", getpass)

    def fake_input(prompt=""):
        out = capsys.readouterr().out
        secrets["root"] = out.split("Secret: ")[1].split()[0]
        return code(secrets["root"], clock["t"])

    monkeypatch.setattr("builtins.input", fake_input)
    assert access.main(["--as", "root", "totp-enroll"]) == 0
    out = capsys.readouterr().out
    assert "Two-factor sign-in is now on" in out and out.count("  ") >= 10
    assert totp_enabled("root", policy)

    clock["t"] += 30
    prompts.clear()
    assert access.main(["--as", "root", "create-group", "team"]) == 0
    assert any(p.startswith("Authentication code") for p in prompts), "an enrolled admin is asked for the code"
    assert access.main(["list"]) == 0
    assert "root owner 2fa" in capsys.readouterr().out

    clock["t"] += 30
    assert access.main(["--as", "root", "totp-disable"]) == 0
    assert not totp_enabled("root", policy)


# --- The sign-in form -----------------------------------------------------------------------------------

def test_the_ui_form_has_a_code_field_after_the_password_and_enforces_it():
    from streamlit.testing.v1 import AppTest

    from rag_lab.security.access import delete_user

    ui = str(Path(__file__).resolve().parent.parent / "ui.py")
    add_user("uitotp", TEST_PASSWORD, role="admin")
    try:
        enrolment = begin_totp_enrollment("uitotp", actor="uitotp")
        confirm_totp_enrollment("uitotp", pyotp.TOTP(enrolment.secret).now(), actor="uitotp")
        totp = pyotp.TOTP(enrolment.secret)

        def attempt(code_value: str):
            app = AppTest.from_file(ui, default_timeout=120).run()
            assert [t.label for t in app.text_input][:2] == ["Username", "Password"], "existing field order is kept"
            app.text_input[0].input("uitotp")
            app.text_input[1].input(TEST_PASSWORD)
            if code_value:
                app.text_input(key="login_code").input(code_value)
            [b for b in app.button if b.label == "Sign in"][0].click().run()
            assert not app.exception, app.exception
            return app

        without = attempt("")
        assert any("Two-factor" in e.value for e in without.error)
        assert "user" not in without.session_state or not without.session_state["user"]

        wrong = attempt("000000")
        assert any("authentication code" in e.value for e in wrong.error)

        accepted = attempt(totp.at(datetime.now() + timedelta(seconds=30)))
        assert accepted.session_state["user"] == "uitotp"
    finally:
        delete_user("uitotp")


def test_the_ui_panel_enrols_shows_recovery_codes_once_and_turns_it_off():
    from streamlit.testing.v1 import AppTest

    from rag_lab.security.access import delete_user

    ui = str(Path(__file__).resolve().parent.parent / "ui.py")
    add_user("uipanel", TEST_PASSWORD, role="admin")
    try:
        app = AppTest.from_file(ui, default_timeout=120).run()
        app.text_input[0].input("uipanel")
        app.text_input[1].input(TEST_PASSWORD)
        [b for b in app.button if b.label == "Sign in"][0].click().run()
        assert not app.exception, app.exception

        app.button(key="totp_setup_btn").click().run()
        assert not app.exception, app.exception
        secret = app.session_state["totp_enrolment"].secret
        assert secret in [c.value for c in app.sidebar.code], "the secret is shown to the person enrolling"

        app.text_input(key="totp_confirm_code").input("000000")
        app.button(key="totp_confirm_btn").click().run()
        assert any("not valid" in e.value for e in app.sidebar.error)
        assert not totp_enabled("uipanel")

        app.text_input(key="totp_confirm_code").input(pyotp.TOTP(secret).now())
        app.button(key="totp_confirm_btn").click().run()
        assert not app.exception, app.exception
        assert totp_enabled("uipanel")
        shown = [c.value for c in app.sidebar.code if "-" in c.value]
        assert shown and len(shown[0].split()) == 10, "the ten recovery codes are shown"

        app.button(key="totp_codes_saved_btn").click().run()
        assert "recovery_codes" not in app.session_state, "and only once"

        recovery = shown[0].split()[0]
        app.text_input(key="totp_current_code").input(recovery)
        app.button(key="totp_off_btn").click().run()
        assert not app.exception, app.exception
        assert not totp_enabled("uipanel")
    finally:
        delete_user("uipanel")
