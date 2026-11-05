"""Key storage: file keys, OS-keychain keys behind a pointer file, migration both ways, and the fallbacks.

No test here may reach the real OS keychain: conftest points the keyring library at its inert backend, and
every test that needs a keychain installs an in-memory one and restores the previous backend afterwards.
"""
import stat
import threading
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest
from cryptography.exceptions import InvalidTag

from rag_lab.config import settings
from rag_lab.security import keystore
from rag_lab.security.keystore import (
    POINTER_PREFIX,
    WRAPPED_PREFIX,
    EnvelopeUnavailableError,
    InvalidKeyError,
    forget_keys,
    key_storage_report,
    load_or_create_key,
    migrate_keys,
    register_envelope_provider,
)
from conftest import TEST_PASSWORD


class MemoryKeyring(keyring.backend.KeyringBackend):
    """An in-memory keychain that counts reads, so tests can see when the process cache saves a round trip."""

    priority = 5  # type: ignore[assignment]

    def __init__(self):
        super().__init__()
        self.items: dict[tuple[str, str], str] = {}
        self.reads = 0

    def set_password(self, servicename, username, password):
        self.items[(servicename, username)] = password

    def get_password(self, servicename, username):
        self.reads += 1
        return self.items.get((servicename, username))

    def delete_password(self, servicename, username):
        if (servicename, username) not in self.items:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.items[(servicename, username)]


@pytest.fixture
def memory_keyring(monkeypatch):
    previous = keyring.get_keyring()
    fake = MemoryKeyring()
    keyring.set_keyring(fake)
    monkeypatch.setattr(settings, "key_store", "keyring")
    yield fake
    keyring.set_keyring(previous)
    keystore._CACHE.clear()


@pytest.fixture
def key_paths(tmp_path, monkeypatch):
    access, data = tmp_path / "keys" / "access.key", tmp_path / "keys" / "data.key"
    monkeypatch.setattr(settings, "access_key_path", str(access))
    monkeypatch.setattr(settings, "data_key_path", str(data))
    # Anything keyed by these temporary keys must also live in temporary stores: an audit event chained with a
    # different key would break the verification of the shared test audit log.
    monkeypatch.setattr(settings, "audit_db_path", str(tmp_path / "default-audit.db"))
    monkeypatch.setattr(settings, "anchor_path", str(tmp_path / "default-anchors.jsonl"))
    monkeypatch.setattr(settings, "access_policy_path", str(tmp_path / "default-policy.json"))
    yield access, data
    keystore._CACHE.clear()


# --- Files, as before ------------------------------------------------------------------------------------

def test_file_keys_are_created_owner_only_and_stable(tmp_path):
    target = tmp_path / "k" / "data.key"
    key = load_or_create_key(str(target))
    assert len(key) == 32 and target.read_bytes() == key
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert load_or_create_key(str(target)) == key


def test_the_test_run_never_uses_the_real_keychain():
    assert settings.key_store == "file"
    assert keystore.keyring_status().usable is False, "conftest points keyring at its inert backend"


def test_concurrent_first_use_creates_one_key_and_one_keychain_entry(tmp_path, memory_keyring):
    target = tmp_path / "race" / "data.key"
    keys: list[bytes] = []

    def worker():
        keys.append(load_or_create_key(str(target)))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(keys)) == 1 and len(keys) == 12
    assert len(memory_keyring.items) == 1, "a losing racer must not leave a second key behind"


# --- The OS keychain -------------------------------------------------------------------------------------

def test_a_new_key_goes_to_the_keychain_behind_a_pointer(tmp_path, memory_keyring):
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))
    blob = target.read_bytes()
    assert blob.startswith(POINTER_PREFIX) and key not in blob
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    (service, account), = memory_keyring.items
    assert service == settings.key_service and account.startswith("data-")
    assert load_or_create_key(str(target)) == key


def test_each_key_gets_its_own_random_account(tmp_path, memory_keyring):
    load_or_create_key(str(tmp_path / "a" / "data.key"))
    load_or_create_key(str(tmp_path / "b" / "data.key"))
    accounts = {account for _, account in memory_keyring.items}
    assert len(accounts) == 2, "two installs must never share a keychain entry"


def test_keychain_keys_are_cached_until_their_pointer_changes(tmp_path, memory_keyring):
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))
    memory_keyring.reads = 0
    for _ in range(50):
        assert load_or_create_key(str(target)) == key
    assert memory_keyring.reads == 0, "the keychain is not asked again on the hot path"

    other = tmp_path / "other.key"
    other_key = load_or_create_key(str(other))
    other.replace(target)  # the pointer file changes underneath the cache
    assert load_or_create_key(str(target)) == other_key != key


def test_a_pointer_whose_entry_is_gone_never_creates_a_replacement_key(tmp_path, memory_keyring):
    target = tmp_path / "data.key"
    load_or_create_key(str(target))
    pointer = target.read_bytes()
    memory_keyring.items.clear()
    keystore._CACHE.clear()
    with pytest.raises(InvalidKeyError, match="does not exist"):
        load_or_create_key(str(target))
    assert target.read_bytes() == pointer
    assert memory_keyring.items == {}


def test_a_pointer_without_a_usable_keychain_is_an_error_not_a_new_key(tmp_path, memory_keyring):
    target = tmp_path / "data.key"
    load_or_create_key(str(target))
    pointer = target.read_bytes()
    keystore._CACHE.clear()
    keyring.set_keyring(keyring.backends.fail.Keyring())
    with pytest.raises(InvalidKeyError, match="OS keychain"):
        load_or_create_key(str(target))
    assert target.read_bytes() == pointer


def test_auto_falls_back_to_a_file_with_a_note_when_there_is_no_system_keychain(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "key_store", "auto")
    keystore._WARNED.clear()
    key = load_or_create_key(str(tmp_path / "data.key"))
    assert (tmp_path / "data.key").read_bytes() == key
    assert "keys are stored in files" in capsys.readouterr().err


def test_auto_ignores_a_backend_that_is_not_a_system_keychain(tmp_path, monkeypatch, memory_keyring):
    """Plaintext file backends (keyrings.alt) and test doubles are not a keychain unless KEY_STORE=keyring says so."""
    monkeypatch.setattr(settings, "key_store", "auto")
    assert keystore.keyring_status().usable is False
    load_or_create_key(str(tmp_path / "data.key"))
    assert memory_keyring.items == {}
    monkeypatch.setattr(keystore, "SYSTEM_BACKENDS", (MemoryKeyring.__module__,))
    assert keystore.keyring_status().usable is True
    load_or_create_key(str(tmp_path / "second" / "data.key"))
    assert len(memory_keyring.items) == 1


def test_required_keychain_that_is_unusable_is_an_error(tmp_path, memory_keyring):
    keyring.set_keyring(keyring.backends.fail.Keyring())
    with pytest.raises(InvalidKeyError, match="KEY_STORE=keyring"):
        load_or_create_key(str(tmp_path / "data.key"))
    assert not (tmp_path / "data.key").exists()


def test_a_keychain_that_refuses_the_write_falls_back_in_auto_and_fails_when_required(tmp_path, monkeypatch, memory_keyring):
    def refuse(*_):
        raise keyring.errors.PasswordSetError("denied")

    monkeypatch.setattr(memory_keyring, "set_password", refuse)
    with pytest.raises(InvalidKeyError, match="refused"):
        load_or_create_key(str(tmp_path / "a.key"))
    assert not (tmp_path / "a.key").exists()

    monkeypatch.setattr(settings, "key_store", "auto")
    monkeypatch.setattr(keystore, "SYSTEM_BACKENDS", (MemoryKeyring.__module__,))
    keystore._WARNED.clear()
    key = load_or_create_key(str(tmp_path / "b.key"))
    assert (tmp_path / "b.key").read_bytes() == key


# --- Migration: real artifacts survive the round trip ----------------------------------------------------

def _make_artifacts(tmp_path):
    """A signed policy, sealed audit rows with a chain and an anchor, sealed cache text and checkpoint state."""
    from rag_lab.security.access import add_user
    from rag_lab.security.audit import anchor, record
    from rag_lab.security.vault import EncryptedSerde, bind_thread, seal_text

    policy, audit, anchors = (str(tmp_path / n) for n in ("policy.json", "audit.db", "anchors.jsonl"))
    add_user("alice", TEST_PASSWORD, admin=True, path=policy)
    for i in range(3):
        record("query_received", workspace_id="ops", session_id=f"s{i}", data={"q": f"question {i}"}, subject="alice",
               path=audit)
    anchor(audit, anchors)
    sealed_text = seal_text("secret answer", b"cache:ops")
    with bind_thread("t1"):
        checkpoint = EncryptedSerde().dumps_typed({"state": "paused"})
    return policy, audit, anchors, sealed_text, checkpoint


def _verify_artifacts(artifacts):
    from rag_lab.security.access import authorize, load_policy, sign_in, verify_token
    from rag_lab.security.audit import list_events, verify_anchors, verify_chain
    from rag_lab.security.vault import EncryptedSerde, bind_thread, open_text

    policy, audit, anchors, sealed_text, checkpoint = artifacts
    assert load_policy(policy)["users"]["alice"]["role"] == "owner", "the policy still verifies"
    result = sign_in("alice", TEST_PASSWORD, policy, audit)
    assert result.ok and verify_token(result.token, path=policy) == "alice", "tokens still verify"
    assert authorize("alice", "ops", "query", policy)
    ok, bad, _ = verify_chain(audit)
    assert ok and bad is None, "the audit chain still verifies"
    assert verify_anchors(audit, anchors)[0], "the anchors still verify"
    assert [e["data"]["q"] for e in list_events(path=audit, event_type="query_received")] == ["question 2", "question 1", "question 0"]
    assert open_text(sealed_text, b"cache:ops") == "secret answer"
    with bind_thread("t1"):
        assert EncryptedSerde(allow_plaintext=False).loads_typed(checkpoint) == {"state": "paused"}
    with bind_thread("other"), pytest.raises(InvalidTag):
        EncryptedSerde().loads_typed(checkpoint)


def test_migration_round_trip_keeps_every_artifact_readable(tmp_path, key_paths, memory_keyring, monkeypatch):
    access, data = key_paths
    monkeypatch.setattr(settings, "key_store", "file")
    artifacts = _make_artifacts(tmp_path)
    raw = {p: p.read_bytes() for p in key_paths}
    assert all(len(b) == 32 for b in raw.values())
    _verify_artifacts(artifacts)

    monkeypatch.setattr(settings, "key_store", "keyring")
    moved = migrate_keys([access, data], "keyring")
    assert [r.action for r in moved] == ["moved", "moved"]
    for path in key_paths:
        assert path.read_bytes().startswith(POINTER_PREFIX)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    stored = sorted(memory_keyring.items.values())
    import base64
    assert stored == sorted(base64.urlsafe_b64encode(b).decode() for b in raw.values()), "the same key bytes moved"
    _verify_artifacts(artifacts)
    assert [r.action for r in migrate_keys([access, data], "keyring")] == ["already", "already"]

    back = migrate_keys([access, data], "file")
    assert [r.action for r in back] == ["moved", "moved"]
    assert {p: p.read_bytes() for p in key_paths} == raw, "the files hold the original keys again"
    assert memory_keyring.items == {}, "no copy of a key is left in the keychain"
    _verify_artifacts(artifacts)
    assert [r.action for r in migrate_keys([access, data], "file")] == ["already", "already"]


def test_migration_skips_a_key_that_does_not_exist_yet(tmp_path, memory_keyring):
    result, = migrate_keys([tmp_path / "none.key"], "keyring")
    assert result.action == "absent" and not (tmp_path / "none.key").exists()


def test_migration_refuses_a_damaged_key_file(tmp_path, memory_keyring):
    damaged = tmp_path / "data.key"
    damaged.write_bytes(b"short")
    with pytest.raises(InvalidKeyError, match="expected 32"):
        migrate_keys([damaged], "keyring")
    assert damaged.read_bytes() == b"short" and memory_keyring.items == {}


def test_migration_into_an_unusable_keychain_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "key_store", "file")
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))
    with pytest.raises(InvalidKeyError, match="Cannot move"):
        migrate_keys([target], "keyring")
    assert target.read_bytes() == key


def test_a_failure_between_the_steps_leaves_a_key_that_still_loads(tmp_path, monkeypatch, memory_keyring):
    monkeypatch.setattr(settings, "key_store", "file")
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))
    monkeypatch.setattr(settings, "key_store", "keyring")

    # 1. The keychain stores something different from what was written: stop before touching the file.
    real_get = keystore._kr_get
    monkeypatch.setattr(keystore, "_kr_get", lambda s, a: b"x" * 32)
    with pytest.raises(InvalidKeyError, match="different value"):
        migrate_keys([target], "keyring")
    assert target.read_bytes() == key and memory_keyring.items == {}
    monkeypatch.setattr(keystore, "_kr_get", real_get)

    # 2. The atomic swap fails: the raw key stays and the entry that was just stored is removed again.
    monkeypatch.setattr(keystore, "_replace", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        migrate_keys([target], "keyring")
    assert target.read_bytes() == key and memory_keyring.items == {}
    monkeypatch.undo()
    monkeypatch.setattr(settings, "key_store", "keyring")

    # 3. The pointer is written but the keychain stops answering: the original key file is restored.
    calls = {"n": 0}
    real_get = keystore._kr_get

    def flaky(service, account):
        calls["n"] += 1
        return real_get(service, account) if calls["n"] == 1 else None

    monkeypatch.setattr(keystore, "_kr_get", flaky)
    with pytest.raises(InvalidKeyError, match="does not exist"):
        migrate_keys([target], "keyring")
    assert target.read_bytes() == key, "the original key file is back"
    assert memory_keyring.items == {}
    monkeypatch.setattr(keystore, "_kr_get", real_get)
    assert load_or_create_key(str(target)) == key


def test_moving_back_to_a_file_survives_a_keychain_that_cannot_delete(tmp_path, memory_keyring, monkeypatch):
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))

    def cannot_delete(*_):
        raise keyring.errors.KeyringError("locked")

    monkeypatch.setattr(memory_keyring, "delete_password", cannot_delete)
    result, = migrate_keys([target], "file")
    assert target.read_bytes() == key and "could not be deleted" in result.detail


def test_forget_removes_the_entries_that_pointers_name(tmp_path, memory_keyring):
    a, b = tmp_path / "a.key", tmp_path / "b.key"
    load_or_create_key(str(a))
    b.write_bytes(b"k" * 32)
    assert forget_keys([a, b, tmp_path / "missing.key"]) == 1
    assert memory_keyring.items == {}


# --- Envelope encryption: the interface and its stub ----------------------------------------------------

class XorProvider:
    """A test double for a KMS: it records the context it was given."""

    name = "testkms"

    def __init__(self):
        self.contexts: list[str] = []

    def wrap(self, key, context):
        self.contexts.append(context)
        return bytes(b ^ 0x5A for b in key)

    def unwrap(self, wrapped, context):
        self.contexts.append(context)
        return bytes(b ^ 0x5A for b in wrapped)


def test_a_registered_provider_wraps_new_keys(tmp_path, monkeypatch):
    provider = XorProvider()
    register_envelope_provider(provider)
    monkeypatch.setattr(settings, "key_envelope", "testkms")
    target = tmp_path / "data.key"
    key = load_or_create_key(str(target))
    assert target.read_bytes().startswith(WRAPPED_PREFIX + b"testkms:") and key not in target.read_bytes()
    keystore._CACHE.clear()
    monkeypatch.setattr(settings, "key_envelope", "")  # reading follows the file, not the setting
    assert load_or_create_key(str(target)) == key
    assert set(provider.contexts) == {"data.key"}, "the key's name is the encryption context"


def test_the_stub_provider_never_stores_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "key_envelope", "stub")
    with pytest.raises(EnvelopeUnavailableError, match="Implement EnvelopeProvider"):
        load_or_create_key(str(tmp_path / "data.key"))
    assert not (tmp_path / "data.key").exists()
    monkeypatch.setattr(settings, "key_envelope", "nonesuch")
    with pytest.raises(EnvelopeUnavailableError, match="not registered"):
        load_or_create_key(str(tmp_path / "data.key"))


def test_a_wrapped_key_without_its_provider_is_an_error_not_a_new_key(tmp_path):
    target = tmp_path / "data.key"
    target.write_bytes(WRAPPED_PREFIX + b"gone:AAAA")
    with pytest.raises(EnvelopeUnavailableError, match="not registered"):
        load_or_create_key(str(target))


# --- Status ------------------------------------------------------------------------------------------------

def test_status_marks_a_headless_fallback_as_advisory_and_a_broken_pointer_as_a_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "key_store", "auto")
    keystore._WARNED.clear()
    target = tmp_path / "data.key"
    target.write_bytes(b"k" * 32)
    (marker, text), = key_storage_report({"data": target})
    assert marker == "--" and "no OS keychain here" in text

    pointer = tmp_path / "access.key"
    pointer.write_bytes(POINTER_PREFIX + b"local-rag-lab:access-deadbeef")
    (marker, text), = key_storage_report({"access": pointer})
    assert marker == "!!" and "OS keychain" in text


def test_status_with_a_working_keychain(tmp_path, memory_keyring):
    moved, fresh = tmp_path / "data.key", tmp_path / "fresh.key"
    load_or_create_key(str(moved))
    lines = dict(key_storage_report({"data": moved, "access": fresh}))
    assert all(marker == "OK" for marker in lines)
    plain = tmp_path / "plain.key"
    plain.write_bytes(b"k" * 32)
    (marker, text), = key_storage_report({"data": plain})
    assert marker == "--" and "make migrate-keys" in text


def test_the_cli_migrates_and_reports(tmp_path, key_paths, memory_keyring, monkeypatch, capsys):
    access, data = key_paths
    monkeypatch.setattr(settings, "key_store", "file")
    load_or_create_key(str(access)), load_or_create_key(str(data))
    monkeypatch.setattr(settings, "key_store", "keyring")
    assert keystore.main(["migrate"]) == 0
    assert "custody" in capsys.readouterr().out
    assert keystore.main(["status"]) == 0
    assert "[OK] access key: OS keychain" in capsys.readouterr().out
    assert keystore.main(["migrate", "--reverse"]) == 0
    assert keystore.main(["forget"]) == 0
    keyring.set_keyring(keyring.backends.fail.Keyring())
    assert keystore.main(["migrate"]) == 1, "an unusable keychain is reported, not raised"
    assert Path(access).read_bytes() != b""
