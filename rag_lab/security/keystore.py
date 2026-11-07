"""Where the local keys live: the OS keychain when one is usable, otherwise owner-only key files.

Two keys protect everything on disk: the access key (signs the policy and session tokens) and the data
key (seals audit payloads, checkpoint state and the cache, and keys the audit chain). Each has a key
file (ACCESS_KEY_PATH, DATA_KEY_PATH) that holds one of three things:

  * the raw 32-byte key (a "key file", what every earlier version wrote),
  * a pointer, `rag-keyring:v1:<service>:<account>`, to an entry in the OS keychain (macOS Keychain,
    Windows Credential Locker, Secret Service/KWallet on Linux), or
  * an envelope-wrapped key, `rag-wrapped:v1:<provider>:<blob>`, unwrapped by a registered provider
    (a Vault/KMS adapter; only the interface and a stub ship, off by default).

The pointer file travels with the data directory and names a random account per key, so two checkouts
never share an entry. KEY_STORE chooses where a *new* key goes: "auto" (keychain when a system backend
is usable, else the file, with a note), "file", or "keyring" (an unusable keychain is an error). An
existing key file stays a file until `make migrate-keys`; nothing moves silently, and a pointer whose
keychain entry is gone is an error, never a reason to mint a replacement key.

`make backup` excludes keys on purpose. A key that lives only in the keychain is lost with the keychain
(an OS reinstall, a keychain reset), and the audit log, checkpoints, cache and policy are unreadable
without it: keep a copy in custody (`make migrate-keys REVERSE=1` puts the key back into its file).
"""
from __future__ import annotations

import base64
import contextlib
import os
import secrets
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:  # advisory file locking (POSIX)
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from rag_lab.config import settings

KEY_BYTES = 32
POINTER_PREFIX = b"rag-keyring:v1:"
WRAPPED_PREFIX = b"rag-wrapped:v1:"
STORES = ("auto", "file", "keyring")
# The platform's own credential stores. In "auto" mode nothing else counts as a keychain: the keyrings.alt
# file backends keep secrets in plaintext, and the inert fail/null backends store nothing.
SYSTEM_BACKENDS = (
    "keyring.backends.macOS",
    "keyring.backends.Windows",
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.libsecret",
)
_INERT_BACKENDS = ("keyring.backends.fail", "keyring.backends.null")


class InvalidKeyError(RuntimeError):
    """A key file is the wrong size or unreadable, or its keychain entry is missing. Keyed checks refuse to run."""


class EnvelopeUnavailableError(InvalidKeyError):
    """An envelope-wrapped key needs a provider that is not registered, or the provider failed."""


# --- Envelope encryption (interface and stub only) -----------------------------------------------------

class EnvelopeProvider(Protocol):
    """Wraps and unwraps a data key under a key held elsewhere (Vault transit, a cloud KMS, an HSM).

    `context` names the key being wrapped ("data.key"); a real provider binds it as the encryption
    context so a wrapped blob cannot be opened as a different key.
    """

    name: str

    def wrap(self, key: bytes, context: str) -> bytes: ...

    def unwrap(self, wrapped: bytes, context: str) -> bytes: ...


class StubEnvelopeProvider:
    """Placeholder that documents the interface. It never holds a key: every call says what is missing."""

    name = "stub"

    def wrap(self, key: bytes, context: str) -> bytes:
        raise EnvelopeUnavailableError(
            "The stub envelope provider wraps nothing. Implement EnvelopeProvider for your Vault/KMS and "
            "register it with rag_lab.security.keystore.register_envelope_provider() at startup."
        )

    def unwrap(self, wrapped: bytes, context: str) -> bytes:
        raise EnvelopeUnavailableError("The stub envelope provider unwraps nothing; register a real provider.")


_PROVIDERS: dict[str, EnvelopeProvider] = {"stub": StubEnvelopeProvider()}


def register_envelope_provider(provider: EnvelopeProvider) -> None:
    _PROVIDERS[provider.name] = provider


def _provider(name: str) -> EnvelopeProvider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise EnvelopeUnavailableError(
            f"Envelope provider '{name}' is not registered (known: {', '.join(sorted(_PROVIDERS))})."
        ) from None


# --- The OS keychain -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyringStatus:
    usable: bool
    backend: str
    reason: str = ""


def keyring_status(system_only: bool | None = None) -> KeyringStatus:
    """Whether a keychain can hold keys here. `system_only` defaults to true unless KEY_STORE=keyring."""
    if system_only is None:
        system_only = settings.key_store != "keyring"
    try:
        import keyring
    except ImportError:
        return KeyringStatus(False, "", "the 'keyring' package is not installed")
    try:
        active = keyring.get_keyring()
    except Exception as exc:  # a backend that fails to initialise
        return KeyringStatus(False, "", f"the keyring library could not select a backend ({exc})")
    module = type(active).__module__
    name = f"{module}.{type(active).__qualname__}"
    if module.startswith(_INERT_BACKENDS) or getattr(active, "priority", 0) <= 0:
        return KeyringStatus(False, name, "no OS keychain is available (headless Linux, a container, or CI)")
    if system_only and not module.startswith(SYSTEM_BACKENDS):
        return KeyringStatus(False, name, f"{name} is not a system keychain")
    return KeyringStatus(True, name)


def _kr_set(service: str, account: str, key: bytes) -> None:
    import keyring
    from keyring.errors import KeyringError

    try:
        keyring.set_password(service, account, base64.urlsafe_b64encode(key).decode("ascii"))
    except KeyringError as exc:
        raise InvalidKeyError(f"The OS keychain refused to store the key: {exc}") from exc


def _kr_get(service: str, account: str) -> bytes | None:
    import keyring
    from keyring.errors import KeyringError

    try:
        value = keyring.get_password(service, account)
    except KeyringError as exc:
        raise InvalidKeyError(f"The OS keychain could not be read: {exc}") from exc
    return base64.urlsafe_b64decode(value) if value else None


def _kr_delete(service: str, account: str) -> None:
    import keyring
    from keyring.errors import KeyringError, PasswordDeleteError

    try:
        keyring.delete_password(service, account)
    except PasswordDeleteError:
        pass  # already gone
    except KeyringError as exc:
        raise InvalidKeyError(f"The OS keychain entry could not be deleted: {exc}") from exc


def _pointer_text(service: str, account: str) -> bytes:
    if not service or any(ch in service for ch in ":\r\n\t "):
        raise InvalidKeyError("KEY_SERVICE must be non-empty and contain no ':' or whitespace.")
    return POINTER_PREFIX + f"{service}:{account}".encode("utf-8")


def _parse_pointer(blob: bytes) -> tuple[str, str]:
    try:
        service, account = blob[len(POINTER_PREFIX):].decode("utf-8").strip().split(":", 1)
    except (UnicodeDecodeError, ValueError):
        raise InvalidKeyError("A key pointer file is malformed.") from None
    return service, account


# --- Reading and creating keys -------------------------------------------------------------------------

# Keychain-backed keys are cached per process: the key is read on every seal, open and MAC, and asking the
# keychain each time is slow and can prompt repeatedly. A cached key is dropped when its pointer file
# changes (new inode, mtime or size). Raw key files are read every time, as before.
_CACHE: dict[str, tuple[tuple[int, int, int], bytes]] = {}
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"Note: {message}", file=sys.stderr)


def _resolve_pointer(target: Path, blob: bytes) -> bytes:
    service, account = _parse_pointer(blob)
    status = keyring_status(system_only=False)
    if not status.usable:
        raise InvalidKeyError(
            f"Key {target} lives in the OS keychain, but {status.reason}. Restore the keychain, or put the "
            "key back into its file where a keychain exists: make migrate-keys REVERSE=1"
        )
    key = _kr_get(service, account)
    if key is None:
        raise InvalidKeyError(
            f"Key {target} points to keychain entry {service}/{account}, which does not exist. "
            "Restore the key from custody. A new key would make existing data unreadable."
        )
    return key


def _resolve_wrapped(target: Path, blob: bytes) -> bytes:
    try:
        provider_name, payload = blob[len(WRAPPED_PREFIX):].decode("ascii").strip().split(":", 1)
        wrapped = base64.urlsafe_b64decode(payload)
    except (UnicodeDecodeError, ValueError):
        raise InvalidKeyError(f"Key file {target} holds a malformed wrapped key.") from None
    try:
        return _provider(provider_name).unwrap(wrapped, target.name)
    except EnvelopeUnavailableError:
        raise
    except Exception as exc:
        raise EnvelopeUnavailableError(f"Envelope provider '{provider_name}' could not unwrap {target}: {exc}") from exc


def _read(target: Path) -> bytes | None:
    try:
        stat = target.stat()
    except FileNotFoundError:
        return None
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(str(target))
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        blob = target.read_bytes()
    except OSError as exc:
        raise InvalidKeyError(f"Key file {target} could not be read: {exc}") from exc
    if blob.startswith(POINTER_PREFIX):
        key = _resolve_pointer(target, blob)
    elif blob.startswith(WRAPPED_PREFIX):
        key = _resolve_wrapped(target, blob)
    else:
        if len(blob) != KEY_BYTES:
            raise InvalidKeyError(
                f"Key file {target} holds {len(blob)} bytes; expected {KEY_BYTES}. "
                "Restore it from custody. A new key would make existing data unreadable."
            )
        return blob
    if len(key) != KEY_BYTES:
        raise InvalidKeyError(
            f"Key {target} resolves to {len(key)} bytes; expected {KEY_BYTES}. Restore it from custody."
        )
    _CACHE[str(target)] = (signature, key)
    return key


@contextlib.contextmanager
def _creation_lock(target: Path) -> Iterator[None]:
    """Serialise creation and migration of one key across threads and processes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    lock_path = target.with_name(f".{target.name}.lock")
    with open(lock_path, "a") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_private(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(target: Path, content: bytes) -> bool:
    """Create the key file atomically with owner-only permissions. False when another process got there first."""
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    _write_private(temporary, content)
    try:
        os.link(temporary, target)
        return True
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _replace(target: Path, content: bytes) -> None:
    """Swap the content of an existing key file atomically (migration). A crash leaves the old or the new file."""
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    _write_private(temporary, content)
    try:
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    _CACHE.pop(str(target), None)


def _new_account(target: Path) -> str:
    return f"{target.stem}-{secrets.token_hex(8)}"


def _store_in_keyring(target: Path, key: bytes) -> bytes:
    """Put a key into the keychain, check it reads back identically, and return the pointer to write."""
    service, account = settings.key_service, _new_account(target)
    pointer = _pointer_text(service, account)
    _kr_set(service, account, key)
    try:
        if _kr_get(service, account) != key:
            raise InvalidKeyError("The OS keychain returned a different value than was stored.")
    except BaseException:
        with contextlib.suppress(InvalidKeyError):
            _kr_delete(service, account)
        raise
    return pointer


def _wrap_for_file(target: Path, key: bytes) -> bytes:
    provider = _provider(settings.key_envelope)
    try:
        wrapped = provider.wrap(key, target.name)
    except EnvelopeUnavailableError:
        raise
    except Exception as exc:
        raise EnvelopeUnavailableError(f"Envelope provider '{provider.name}' could not wrap {target}: {exc}") from exc
    return WRAPPED_PREFIX + provider.name.encode("ascii") + b":" + base64.urlsafe_b64encode(wrapped)


def _create(target: Path) -> bytes:
    """Make a new key in the configured store. The key is stored before any file says it exists."""
    with _creation_lock(target):
        existing = _read(target)
        if existing is not None:
            return existing
        if settings.key_store not in STORES:
            raise InvalidKeyError(f"KEY_STORE must be one of {', '.join(STORES)}; got '{settings.key_store}'.")
        key = secrets.token_bytes(KEY_BYTES)
        content = key
        pointer_account: tuple[str, str] | None = None
        if settings.key_envelope:
            content = _wrap_for_file(target, key)
        elif settings.key_store != "file":
            status = keyring_status()
            try:
                if not status.usable:
                    raise InvalidKeyError(f"KEY_STORE={settings.key_store} but {status.reason}.")
                content = _store_in_keyring(target, key)
                pointer_account = _parse_pointer(content)
            except InvalidKeyError as exc:
                if settings.key_store == "keyring":
                    raise
                _warn_once(f"keys are stored in files ({exc}). Key files are owner-only; see `make status`.")
        if not _publish(target, content):  # only possible where file locking is unavailable
            if pointer_account is not None:
                with contextlib.suppress(InvalidKeyError):
                    _kr_delete(*pointer_account)
    loaded = _read(target)
    assert loaded is not None
    return loaded


def load_or_create_key(path: str | None = None) -> bytes:
    """Return a 32-byte key, creating it in the configured store on first use.

    Creation is serialised and atomic: a reader sees no key or a complete one, never a partial file, and a
    keychain-backed key is stored before the pointer file that names it appears. A key of any other length
    is refused: an empty HMAC key would make every signature forgeable.
    """
    target = Path(path or settings.data_key_path)
    key = _read(target)
    return key if key is not None else _create(target)


def _bytes_or_none(target: Path) -> bytes | None:
    try:
        return target.read_bytes()
    except OSError:
        return None


# --- Migration between stores --------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationResult:
    path: Path
    action: str  # "moved", "already", "absent"
    detail: str


def _kind(blob: bytes) -> str:
    if blob.startswith(POINTER_PREFIX):
        return "keychain"
    if blob.startswith(WRAPPED_PREFIX):
        return "wrapped"
    if len(blob) == KEY_BYTES:
        return "file"
    raise InvalidKeyError(f"holds {len(blob)} bytes; expected {KEY_BYTES}")


def _migrate_one(target: Path, to: str) -> MigrationResult:
    with _creation_lock(target):
        if not target.exists():
            return MigrationResult(target, "absent", "no key yet; it is created in the configured store on first use")
        blob = target.read_bytes()
        try:
            kind = _kind(blob)
        except InvalidKeyError as exc:
            raise InvalidKeyError(f"Key file {target} {exc}. Restore it from custody first.") from None
        if (to, kind) in (("keyring", "keychain"), ("file", "file")):
            return MigrationResult(target, "already", f"already in the {'OS keychain' if to == 'keyring' else 'key file'}")
        _CACHE.pop(str(target), None)
        key = _read(target)  # resolves a pointer or a wrapped key; raises if it cannot be read
        assert key is not None
        if to == "keyring":
            status = keyring_status()
            if not status.usable:
                raise InvalidKeyError(f"Cannot move {target} into the OS keychain: {status.reason}.")
            pointer = _store_in_keyring(target, key)  # stored and read back before the file changes
            service, account = _parse_pointer(pointer)
            try:
                _replace(target, pointer)
                if _read(target) != key:
                    raise InvalidKeyError("The key read back through its pointer differs from the original.")
            except BaseException:
                if _bytes_or_none(target) != blob:  # a pointer was written but does not work: restore the original
                    _replace(target, blob)
                with contextlib.suppress(InvalidKeyError):
                    _kr_delete(service, account)
                raise
            return MigrationResult(target, "moved", f"now in the OS keychain ({status.backend}); {target.name} holds a pointer")
        _replace(target, key)
        if _read(target) != key:
            raise InvalidKeyError("The key read back from its file differs from the original.")
        note = "now in the key file"
        if kind == "keychain":
            service, account = _parse_pointer(blob)
            try:
                _kr_delete(service, account)
            except InvalidKeyError as exc:
                note += f" (the old keychain entry {service}/{account} could not be deleted: {exc})"
        return MigrationResult(target, "moved", note)


def migrate_keys(paths: list[Path], to: str) -> list[MigrationResult]:
    """Move each key into the OS keychain (`to="keyring"`) or back into its file (`to="file"`).

    The key is stored in its new home and read back before the file changes, and the file is swapped
    atomically, so an interruption at any point leaves a key that still loads.
    """
    if to not in ("keyring", "file"):
        raise ValueError("to must be 'keyring' or 'file'")
    return [_migrate_one(path, to) for path in paths]


def forget_keys(paths: list[Path]) -> int:
    """Delete the keychain entries that key files point to (factory reset). Returns how many were removed."""
    removed = 0
    for target in paths:
        blob = _bytes_or_none(target)
        if blob is not None and blob.startswith(POINTER_PREFIX):
            _kr_delete(*_parse_pointer(blob))
            removed += 1
    return removed


# --- Reporting -----------------------------------------------------------------------------------------

def key_storage_report(paths: dict[str, Path]) -> list[tuple[str, str]]:
    """(marker, text) lines for `make status`. A headless fallback to files is advisory, a broken pointer is not."""
    lines: list[tuple[str, str]] = []
    status = keyring_status()
    for name, target in paths.items():
        blob = _bytes_or_none(target)
        if blob is None:
            if settings.key_envelope:
                where = "its key file (envelope-wrapped: KEY_ENVELOPE is set)"
            elif settings.key_store != "file" and status.usable:
                where = "the OS keychain"
            else:
                where = "its key file"
            lines.append(("OK", f"{name} key: not created yet (it will be created in {where})"))
        elif blob.startswith(POINTER_PREFIX):
            try:
                _read(target)
                lines.append(("OK", f"{name} key: OS keychain ({status.backend or 'backend unavailable'})"))
            except InvalidKeyError as exc:
                lines.append(("!!", f"{name} key: {exc}"))
        elif blob.startswith(WRAPPED_PREFIX):
            try:
                _read(target)
                lines.append(("OK", f"{name} key: envelope-wrapped"))
            except InvalidKeyError as exc:
                lines.append(("!!", f"{name} key: {exc}"))
        else:
            if status.usable and settings.key_store != "file":
                advice = "run 'make migrate-keys' to move it into the OS keychain"
            elif settings.key_store == "file":
                advice = "KEY_STORE=file"
            else:
                advice = f"no OS keychain here: {status.reason}"
            lines.append(("--", f"{name} key: key file, owner-only ({advice})"))
    return lines


def _key_paths() -> dict[str, Path]:
    return {"access": Path(settings.access_key_path), "data": Path(settings.data_key_path)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Where the access and data keys are stored.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show where each key lives")
    migrate = sub.add_parser("migrate", help="Move the keys into the OS keychain, or back into their files")
    migrate.add_argument("--reverse", action="store_true", help="Move keys from the keychain back into their files")
    sub.add_parser("forget", help="Delete the keychain entries the key files point to (used by factory-reset)")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            for marker, text in key_storage_report(_key_paths()):
                print(f"[{marker}] {text}")
        elif args.command == "migrate":
            to = "file" if args.reverse else "keyring"
            for result in migrate_keys(list(_key_paths().values()), to):
                print(f"{result.path.name}: {result.detail}")
            if to == "keyring":
                print("Keep a copy of each key in custody: 'make backup' excludes keys, and keys held only in the "
                      "keychain are lost with it. 'make migrate-keys REVERSE=1' puts them back into their files.")
        else:
            print(f"Removed {forget_keys(list(_key_paths().values()))} keychain entr(ies).")
    except InvalidKeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
