#!/usr/bin/env python3
"""Local identity, sign-in, roles, groups, and workspace access control.

Users, scrypt password hashes, roles, groups, per-workspace grants, and the default role live in one
JSON file (ACCESS_POLICY_PATH). The file carries an HMAC tag made with a local signing key
(ACCESS_KEY_PATH); if it is edited outside this module, every check fails closed.

A role is a named bundle of the workspace permissions (ROLE_PERMISSIONS); "owner" and "admin" also carry
administration (users, grants, groups, workspaces, indexing). Permissions on one workspace resolve in
this order, most specific first:
  1. the account's explicit grant on that workspace, then its explicit grant on all workspaces
     (`grant <user> all`); an empty list is an explicit denial,
  2. the grants of the account's groups on that workspace (union; any group denial wins),
  3. the account's own role,
  4. the policy's default role ("trusted" on a fresh install: query + auto_approve).
So out of the box every account can chat on every workspace without review. "review" is never part of
the default role: reviewers are appointed. Clearance is separate from roles. Only administrators may
change accounts, roles, groups, grants, defaults, or clearances; only an owner may assign or remove the
owner role, and the last owner stays.

Policy files from version 1 and 2 are migrated to version 3 on first load, after their old tag verifies,
with a backup of the old file and no change to anyone's effective access.

Identity crosses into the pipeline as an HMAC-signed session token, never as a bare name.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeVar

import pyotp
from cryptography.exceptions import InvalidTag

try:  # advisory file locking for the policy file (POSIX)
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from rag_lab.security.audit import count_events_since
from rag_lab.security.audit import record as audit_record
from rag_lab.config import settings
from rag_lab.security.vault import load_or_create_key, open_text, seal_text
from rag_lab.storage.workspace_manager import sanitize_workspace_id

PERMISSIONS = ("query", "review", "auto_approve")
# Roles are named bundles of the workspace permissions. "owner" and "admin" also administer the system.
ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "owner": ("auto_approve", "query", "review"),
    "admin": ("auto_approve", "query", "review"),
    "reviewer": ("query", "review"),
    "trusted": ("auto_approve", "query"),
    "member": ("query",),
    "none": (),
}
ROLES = tuple(ROLE_PERMISSIONS)
ADMIN_ROLES = ("owner", "admin")
# Roles a group grant or the policy default may hold: never administration.
WORKSPACE_ROLES = ("reviewer", "trusted", "member", "none")
DEFAULT_ROLE = "trusted"
# What every account holds on every workspace until an administrator says otherwise.
DEFAULT_PERMISSIONS = ROLE_PERMISSIONS[DEFAULT_ROLE]
POLICY_VERSION = 3
CLEARANCE_LEVELS = ("public", "internal", "confidential", "secret")
DEFAULT_CLEARANCE = "public"
MIN_PASSWORD_LENGTH = 12
LOCKOUT_MAX_FAILURES = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60
TOKEN_TTL_SECONDS = 8 * 60 * 60
IDLE_TIMEOUT_SECONDS = 30 * 60
TOTP_ISSUER = "Local RAG Lab"
TOTP_WINDOW = 1  # accept the code of the previous and the next 30-second step as well, for clock drift
RECOVERY_CODE_COUNT = 10

_USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{1,63}")
_WORKSPACE_ID = re.compile(r"[a-z0-9_]{1,48}")
_SCRYPT_PARAMS = {"n": 2**17, "r": 8, "p": 1}
_SCRYPT_MAXMEM = 256 * 1024 * 1024
_DENYLIST_PATH = Path(__file__).resolve().parent / "common_passwords.txt"


_T = TypeVar("_T")


class PolicyIntegrityError(RuntimeError):
    """The access policy file was changed outside access.py, or its signing key is missing or wrong."""


@dataclass(frozen=True)
class SignInResult:
    ok: bool
    username: str = ""
    token: str = ""
    reason: str = ""
    retry_after_minutes: int = 0

    def describe(self) -> str:
        if self.ok:
            return "Signed in."
        if self.reason == "locked":
            return (
                f"Too many failed sign-ins for '{self.username}'. "
                f"Try again in about {self.retry_after_minutes} minutes."
            )
        if self.reason == "totp_required":
            return "Two-factor sign-in is on for this account: enter the code from your authenticator app, or a recovery code."
        if self.reason == "invalid_code":
            return "Sign-in failed: the authentication code is wrong or was already used."
        return "Sign-in failed: the account name or password is wrong."


# --- Password hashing -------------------------------------------------------

def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=_SCRYPT_MAXMEM
    )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = _SCRYPT_PARAMS["n"], _SCRYPT_PARAMS["r"], _SCRYPT_PARAMS["p"]
    digest = _scrypt(password, salt, n, r, p, 32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = _scrypt(password, bytes.fromhex(salt_hex), int(n), int(r), int(p), len(digest_hex) // 2)
    return hmac.compare_digest(digest.hex(), digest_hex)


def needs_rehash(stored: str) -> bool:
    """True when a stored hash uses older scrypt parameters and should be upgraded at the next sign-in."""
    try:
        _, n, r, p, _, _ = stored.split("$")
    except ValueError:
        return True
    return (int(n), int(r), int(p)) != (_SCRYPT_PARAMS["n"], _SCRYPT_PARAMS["r"], _SCRYPT_PARAMS["p"])


@lru_cache(maxsize=1)
def _timing_equalizer() -> str:
    """Checked against unknown usernames so sign-in timing does not reveal which accounts exist."""
    return hash_password("timing-equalizer-not-a-password")


@lru_cache(maxsize=1)
def _common_passwords() -> frozenset[str]:
    lines = _DENYLIST_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip().lower() for line in lines if line.strip() and not line.startswith("#"))


def check_password_strength(password: str, username: str = "") -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Passwords must be at least {MIN_PASSWORD_LENGTH} characters.")
    lowered = password.lower()
    if lowered in _common_passwords():
        raise ValueError("That password is too common. Choose a less predictable one.")
    if username and username.lower() in lowered:
        raise ValueError("The password must not contain the account name.")


# --- Signing key, policy integrity, and session tokens ----------------------

def _signing_key(path: str | None = None) -> bytes:
    """The local signing key for tokens and the policy tag. Created with owner-only permissions on first use."""
    return load_or_create_key(path or settings.access_key_path)


def _role_for(permissions: list[str] | tuple[str, ...], choices: tuple[str, ...] = WORKSPACE_ROLES) -> str | None:
    """The role whose bundle is exactly these permissions, if one exists among `choices`."""
    wanted = sorted(set(permissions))
    return next((r for r in choices if sorted(ROLE_PERMISSIONS[r]) == wanted), None)


def _canonical_defaults(defaults: Any) -> dict[str, Any]:
    """The policy default as {"role": r}, or {"permissions": [...]} for a custom set no role matches."""
    if isinstance(defaults, dict) and set(defaults) == {"role"}:
        if defaults["role"] not in WORKSPACE_ROLES:
            raise PolicyIntegrityError("Access policy default role is malformed. Access is refused until it is restored.")
        return {"role": defaults["role"]}
    perms = defaults.get("permissions") if isinstance(defaults, dict) and set(defaults) == {"permissions"} else None
    if not isinstance(perms, list) or any(p not in PERMISSIONS for p in perms):
        raise PolicyIntegrityError("Access policy defaults are malformed. Access is refused until it is restored.")
    role = _role_for(perms)
    return {"role": role} if role else {"permissions": sorted(set(perms))}


def _canonical_groups(groups: Any) -> dict[str, Any]:
    if not isinstance(groups, dict):
        raise PolicyIntegrityError("Access policy groups are malformed. Access is refused until it is restored.")
    for info in groups.values():
        grants = info.get("grants") if isinstance(info, dict) else None
        if not isinstance(grants, dict) or any(r not in WORKSPACE_ROLES for r in grants.values()):
            raise PolicyIntegrityError("Access policy groups are malformed. Access is refused until it is restored.")
    return groups


def _legacy_mac(users: dict[str, Any], defaults: dict[str, Any] | None) -> str:
    """The tag of policy versions 1 (users only) and 2 (users and permission defaults). Read only to migrate."""
    if defaults is None:
        body = json.dumps(users, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(_signing_key(), b"policy:" + body, hashlib.sha256).hexdigest()
    body = json.dumps({"users": users, "defaults": defaults}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_signing_key(), b"policy2:" + body, hashlib.sha256).hexdigest()


def _policy_mac(users: dict[str, Any], defaults: dict[str, Any], groups: dict[str, Any]) -> str:
    """HMAC over users, groups, and the default (policy version 3), in its own domain."""
    body = json.dumps(
        {"users": users, "defaults": defaults, "groups": groups}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hmac.new(_signing_key(), b"policy3:" + body, hashlib.sha256).hexdigest()


def new_policy() -> dict[str, Any]:
    """The policy of a fresh install: no accounts, no groups, and the out-of-the-box default role."""
    return {"version": POLICY_VERSION, "users": {}, "groups": {}, "defaults": {"role": DEFAULT_ROLE}}


def default_permissions_of(policy: dict[str, Any]) -> list[str]:
    defaults = policy.get("defaults") or {}
    if "role" in defaults:
        return sorted(ROLE_PERMISSIONS[defaults["role"]])
    return sorted(defaults.get("permissions", []))


def migrate_policy(policy: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Convert a verified version 1 or 2 policy to version 3 without changing anyone's effective access.

    Admins become role "owner" (the first, alphabetically) or "admin". Where that role's bundle differs
    from what the account held through the old defaults, the old access is pinned as the account's
    explicit grant on all workspaces. Returns the new policy and one note per pinned account.
    """
    legacy = policy.get("defaults")
    # A version 1 file predates defaults and kept its strict meaning: no access beyond explicit grants.
    defaults = _canonical_defaults(legacy) if legacy is not None else {"role": "none"}
    old_defaults = default_permissions_of({"defaults": defaults})
    admins = sorted(name for name, info in policy["users"].items() if info.get("admin"))
    users: dict[str, Any] = {}
    notes: list[str] = []
    for name, info in policy["users"].items():
        user = {k: v for k, v in info.items() if k != "admin"}
        if name in admins:
            user["role"] = "owner" if name == admins[0] else "admin"
            if "default_permissions" not in user and old_defaults != sorted(ROLE_PERMISSIONS[user["role"]]):
                user["default_permissions"] = list(old_defaults)
                notes.append(
                    f"{name} ({user['role']}): access pinned to {'/'.join(old_defaults) or 'none'} on all workspaces"
                )
        users[name] = user
    return {"version": POLICY_VERSION, "users": users, "groups": {}, "defaults": defaults}, notes


def _read_policy_file(target: Path) -> dict[str, Any]:
    try:
        with open(target, encoding="utf-8") as handle:
            policy = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyIntegrityError(
            f"Access policy {target} could not be parsed. Access is refused until it is restored."
        ) from exc
    if not isinstance(policy, dict) or not isinstance(policy.get("users"), dict):
        raise PolicyIntegrityError(
            f"Access policy {target} is malformed. Access is refused until it is restored."
        )
    return policy


def _verified(policy: dict[str, Any], target: Path) -> dict[str, Any]:
    """Check the tag of a version 3 policy, or of a legacy one, and return it with canonical fields."""
    failed = PolicyIntegrityError(
        f"Access policy {target} failed its integrity check. Access is refused until it is restored."
    )
    version = policy.get("version", 1)
    users = policy["users"]
    if version == POLICY_VERSION:
        defaults = _canonical_defaults(policy.get("defaults"))
        groups = _canonical_groups(policy.get("groups", {}))
        if not hmac.compare_digest(str(policy.get("integrity", "")), _policy_mac(users, defaults, groups)):
            raise failed
        return {**policy, "defaults": defaults, "groups": groups}
    if version not in (1, 2):
        raise PolicyIntegrityError(
            f"Access policy {target} has unknown version {version!r}. Access is refused until it is restored."
        )
    legacy = policy.get("defaults")
    if legacy is not None:
        perms = legacy.get("permissions") if isinstance(legacy, dict) else None
        if not isinstance(perms, list) or any(p not in PERMISSIONS for p in perms):
            raise PolicyIntegrityError("Access policy defaults are malformed. Access is refused until it is restored.")
        legacy = {"permissions": sorted(set(perms))}
    if not hmac.compare_digest(str(policy.get("integrity", "")), _legacy_mac(users, legacy)):
        raise failed
    return policy


def load_policy(path: str | None = None) -> dict[str, Any]:
    target = Path(path or settings.access_policy_path)
    if not target.is_file():
        return new_policy()
    policy = _verified(_read_policy_file(target), target)
    if policy.get("version") == POLICY_VERSION:
        return policy
    with _policy_lock(path):  # another process may have migrated it meanwhile: re-read under the lock
        policy = _verified(_read_policy_file(target), target)
        if policy.get("version") == POLICY_VERSION:
            return policy
        old_version = policy.get("version", 1)
        backup = target.with_name(f"{target.name}.v{old_version}-{datetime.now(UTC):%Y%m%dT%H%M%S%f}.bak")
        shutil.copy2(target, backup)
        os.chmod(backup, 0o600)
        migrated, notes = migrate_policy(policy)
        save_policy(migrated, path)
    audit_record("policy_migrated", workspace_id="", session_id="",
                 data={"from": old_version, "to": POLICY_VERSION, "backup": backup.name, "pinned": notes})
    print(f"Access policy migrated from version {old_version} to {POLICY_VERSION}; backup at {backup}.",
          file=sys.stderr)
    for note in notes:
        print(f"  {note}", file=sys.stderr)
    return load_policy(path)


def save_policy(policy: dict[str, Any], path: str | None = None) -> None:
    """Sign and atomically write a policy. A version 1 or 2 dict (older seed code) is migrated first."""
    target = Path(path or settings.access_policy_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if policy.get("version", 1) != POLICY_VERSION:
        policy, _ = migrate_policy(policy)
    defaults = _canonical_defaults(policy.get("defaults", {"role": DEFAULT_ROLE}))
    groups = _canonical_groups(policy.get("groups", {}))
    users = policy.get("users", {})
    policy = {**policy, "version": POLICY_VERSION, "defaults": defaults, "groups": groups,
              "integrity": _policy_mac(users, defaults, groups)}
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".access-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


_LOCKS_HELD = threading.local()


@contextlib.contextmanager
def _policy_lock(path: str | None = None):
    """Serialize read-modify-write cycles on the policy file across threads and processes.

    Re-entrant within a thread, so a load that migrates the file can run inside an update.
    """
    target = Path(path or settings.access_policy_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    held: set[str] = _LOCKS_HELD.__dict__.setdefault("paths", set())
    key = str(target.resolve())
    if fcntl is None or key in held:  # Windows has no flock; a held lock is re-entered
        yield
        return
    lock_path = target.with_name(target.name + ".lock")
    with open(lock_path, "a") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
            fcntl.flock(handle, fcntl.LOCK_UN)


def _update(path: str | None, mutate: Callable[[dict[str, Any]], _T]) -> _T:
    """Load the policy, apply `mutate(policy)`, and save it, all under the policy lock."""
    with _policy_lock(path):
        policy = load_policy(path)
        result = mutate(policy)
        save_policy(policy, path)
        return result


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _token_mac(body: str) -> str:
    return _b64encode(hmac.new(_signing_key(), b"token:" + body.encode("ascii"), hashlib.sha256).digest())


_LAST_POLICY_BY_UID: dict[str, str] = {}


def _token_version(user: dict[str, Any] | None) -> str:
    """Identifies one incarnation of an account. It changes when sessions are revoked or the name is reused."""
    if not user or not user.get("uid"):
        return ""
    return f"{user['uid']}.{int(user.get('token_version', 0))}"


def issue_token(username: str, now: float | None = None, path: str | None = None) -> str:
    issued = int(now if now is not None else time.time())
    try:
        user = load_policy(path)["users"].get(username)
        version = _token_version(user)
        if user and user.get("uid"):
            _LAST_POLICY_BY_UID[str(user["uid"])] = str(Path(path or settings.access_policy_path).resolve())
    except PolicyIntegrityError:
        version = ""
    payload = json.dumps(
        {"sub": username, "iat": issued, "exp": issued + TOKEN_TTL_SECONDS, "tv": version},
        separators=(",", ":"),
    )
    body = _b64encode(payload.encode("utf-8"))
    return f"{body}.{_token_mac(body)}"


def verify_token(token: str, now: float | None = None, path: str | None = None) -> str:
    """Return the account a valid, unexpired, unrevoked token was issued to, or "" for anything else.

    A token stops working when its account is deleted, recreated under the same name, or has its
    sessions revoked (a password reset does that).
    """
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(_token_mac(body), signature):
            return ""
        payload = json.loads(_b64decode(body))
        current = now if now is not None else time.time()
        if int(payload["exp"]) <= current:
            return ""
        subject = str(payload["sub"])
        user = load_policy(path)["users"].get(subject)
        if user is None and path is None:
            uid_part = str(payload.get("tv", "")).split(".", 1)[0]
            origin_path = _LAST_POLICY_BY_UID.get(uid_part)
            if origin_path and Path(origin_path).exists():
                user = load_policy(origin_path)["users"].get(subject)
        if user is None or str(payload.get("tv", "")) != _token_version(user):
            return ""
        return subject
    except (ValueError, KeyError, TypeError, AttributeError, PolicyIntegrityError):
        return ""


def is_idle_expired(last_seen: float, now: float, timeout_seconds: int = IDLE_TIMEOUT_SECONDS) -> bool:
    return now - last_seen > timeout_seconds


# --- Accounts and sign-in ---------------------------------------------------

def authenticate(username: str, password: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username.strip().lower())
    stored = user.get("password_hash", "") if user else ""
    if not stored:
        verify_password(password, _timing_equalizer())
        return False
    return verify_password(password, stored)


def sign_in(
    username: str,
    password: str,
    path: str | None = None,
    audit_path: str | None = None,
    code: str | None = None,
    code_prompt: Callable[[], str] | None = None,
) -> SignInResult:
    """Check a password (and the second factor, when the account has one), apply the lockout, issue a token.

    An enrolled account needs an authentication code or a recovery code. It is asked for only after the
    password verifies: with no `code`, `code_prompt()` supplies one (a CLI), else the result says
    "totp_required" and nothing is counted. A wrong code is a failed sign-in and counts toward the lockout.
    Failed attempts are written to the audit log; the lockout count is read back from it.
    """
    name = (username or "").strip().lower()[:64]
    if name:
        since = datetime.now(UTC) - timedelta(seconds=LOCKOUT_WINDOW_SECONDS)
        recent = count_events_since("sign_in_failed", name, since, path=audit_path)
        if recent >= LOCKOUT_MAX_FAILURES:
            audit_record("sign_in_blocked", workspace_id="", session_id="", data={"username": name},
                         subject=name, path=audit_path)
            return SignInResult(
                ok=False, username=name, reason="locked", retry_after_minutes=LOCKOUT_WINDOW_SECONDS // 60
            )

    if not name or not authenticate(name, password, path):
        audit_record("sign_in_failed", workspace_id="", session_id="", data={"username": name},
                     subject=name, path=audit_path)
        return SignInResult(ok=False, username=name, reason="invalid")

    user = load_policy(path)["users"][name]
    if user.get("totp"):  # enforced for as long as the enrolment exists, whatever the account's role is now
        supplied = code if code else (code_prompt() if code_prompt else "")
        if not supplied.strip():
            return SignInResult(ok=False, username=name, reason="totp_required")
        method, remaining = _consume_second_factor(name, supplied, path)
        if not method:
            audit_record("sign_in_failed", workspace_id="", session_id="",
                         data={"username": name, "stage": "second_factor"}, subject=name, path=audit_path)
            return SignInResult(ok=False, username=name, reason="invalid_code")
        if method == "recovery":
            audit_record("recovery_code_used", workspace_id="", session_id="",
                         data={"username": name, "remaining": remaining}, subject=name, path=audit_path)

    if needs_rehash(load_policy(path)["users"][name]["password_hash"]):
        def upgrade(policy: dict[str, Any]) -> None:
            policy["users"][name]["password_hash"] = hash_password(password)

        _update(path, upgrade)

    audit_record("sign_in_succeeded", workspace_id="", session_id="", data={"username": name},
                 subject=name, path=audit_path)
    return SignInResult(ok=True, username=name, token=issue_token(name, path=path))


# --- Two-factor sign-in (TOTP) ----------------------------------------------------------------

@dataclass(frozen=True)
class TotpEnrollment:
    secret: str
    uri: str


def _now() -> float:
    return time.time()


def _match_totp_step(secret: str, code: str) -> int | None:
    """The 30-second step a code belongs to, within the drift window, or None.

    pyotp's TOTP.verify() says only yes or no; replay protection needs the step, to refuse any step that is
    not newer than the last one accepted. Every step in the window is compared, so timing does not reveal
    which one matched.
    """
    totp = pyotp.TOTP(secret)
    current = int(_now() // totp.interval)
    supplied = code.encode("utf-8")
    matched: int | None = None
    for step in range(current - TOTP_WINDOW, current + TOTP_WINDOW + 1):
        if hmac.compare_digest(supplied, totp.generate_otp(step).encode("ascii")):
            matched = step
    return matched


def _totp_aad(name: str, user: dict[str, Any]) -> bytes:
    """Binds a sealed secret to its account, so it cannot be copied into another account's record."""
    return f"totp:{user.get('uid', '')}:{name}".encode("utf-8")


def _recovery_tag(code: str) -> str:
    cleaned = re.sub(r"[^0-9a-z]", "", code.lower())
    return hmac.new(_signing_key(), b"recovery:" + cleaned.encode("ascii", "ignore"), hashlib.sha256).hexdigest()


def _new_recovery_codes() -> list[str]:
    codes = []
    for _ in range(RECOVERY_CODE_COUNT):
        raw = secrets.token_hex(6)
        codes.append("-".join(raw[i:i + 4] for i in range(0, 12, 4)))
    return codes


def totp_enabled(username: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username)
    return bool(user and user.get("totp"))


def recovery_codes_left(username: str, path: str | None = None) -> int:
    user = load_policy(path)["users"].get(username) or {}
    return len((user.get("totp") or {}).get("recovery", []))


def _consume_second_factor(name: str, code: str, path: str | None) -> tuple[str, int]:
    """Check a code and spend it, all under the policy lock so two sign-ins cannot both use one code.

    A TOTP code is accepted once: its time step must be newer than the last accepted one. A recovery code
    is removed when used. Returns ("totp" | "recovery" | "", recovery codes left).
    """
    cleaned = code.strip()

    def mutate(policy: dict[str, Any]) -> tuple[str, int]:
        user = policy["users"][name]
        enrolment = user["totp"]
        try:
            secret = open_text(enrolment["secret"], _totp_aad(name, user), allow_plaintext=False)
        except InvalidTag:
            raise PolicyIntegrityError(
                f"The two-factor secret of '{name}' cannot be opened (the data key changed, or the policy was "
                "edited). Restore the data key from custody; an owner can reset the account's two-factor sign-in."
            ) from None
        step = _match_totp_step(secret, cleaned.replace(" ", ""))
        if step is not None and step > int(enrolment.get("last_step", 0)):
            enrolment["last_step"] = int(step)
            return "totp", len(enrolment.get("recovery", []))
        tag = _recovery_tag(cleaned)
        for stored in list(enrolment.get("recovery", [])):
            if hmac.compare_digest(stored, tag):
                enrolment["recovery"].remove(stored)
                return "recovery", len(enrolment["recovery"])
        return "", len(enrolment.get("recovery", []))

    return _update(path, mutate)


def begin_totp_enrollment(username: str, path: str | None = None, actor: str | None = None) -> TotpEnrollment:
    """Start two-factor enrolment for the signed-in owner or administrator themselves.

    The secret is kept sealed as pending until confirm_totp_enrollment() sees a valid code from it, so a
    secret that was never loaded into an authenticator cannot lock the account out.
    """
    name = _normalize_username(username)
    if actor is None or _normalize_username(actor) != name:
        raise PermissionError("Two-factor sign-in is enrolled by the account itself, never on its behalf.")
    secret = pyotp.random_base32()

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        if role_of(user) not in ADMIN_ROLES:
            raise PermissionError("Two-factor sign-in is for owners and administrators.")
        if user.get("totp"):
            raise ValueError("Two-factor sign-in is already on. Turn it off first to enrol a new device.")
        user["totp_pending"] = seal_text(secret, _totp_aad(name, user))

    _update(path, mutate)
    _audit("totp_enrolment_started", name, {"username": name})
    return TotpEnrollment(secret, pyotp.TOTP(secret).provisioning_uri(name=name, issuer_name=TOTP_ISSUER))


def confirm_totp_enrollment(
    username: str, code: str, path: str | None = None, actor: str | None = None
) -> list[str]:
    """Turn two-factor sign-in on once a code from the pending secret checks out. Returns the recovery codes (shown once)."""
    name = _normalize_username(username)
    if actor is None or _normalize_username(actor) != name:
        raise PermissionError("Two-factor sign-in is enrolled by the account itself, never on its behalf.")
    codes = _new_recovery_codes()

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        pending = user.get("totp_pending")
        if not pending:
            raise ValueError("No enrolment is in progress. Start one first.")
        try:
            secret = open_text(pending, _totp_aad(name, user), allow_plaintext=False)
        except InvalidTag:
            raise ValueError("The pending enrolment cannot be opened. Start it again.") from None
        step = _match_totp_step(secret, code.strip().replace(" ", ""))
        if step is None:
            raise ValueError("That code is not valid. Check the device clock and try again.")
        del user["totp_pending"]
        user["totp"] = {
            "secret": pending,
            "last_step": int(step),
            "recovery": [_recovery_tag(c) for c in codes],
            "enrolled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    _update(path, mutate)
    _audit("totp_enrolled", name, {"username": name, "recovery_codes": len(codes)})
    return codes


def regenerate_recovery_codes(
    username: str, code: str, path: str | None = None, actor: str | None = None, verified: bool = False
) -> list[str]:
    """Replace the account's recovery codes. Needs a current authentication code (a recovery code works too).

    `verified` is for a command that has just signed the account in with its second factor (the CLI): that
    code is already spent for this time step, and asking again would only force a 30-second wait.
    """
    name = _normalize_username(username)
    if actor is None or _normalize_username(actor) != name:
        raise PermissionError("Recovery codes are regenerated by the account itself.")
    if not totp_enabled(name, path):
        raise ValueError("Two-factor sign-in is not on for this account.")
    if not verified and not _consume_second_factor(name, code, path)[0]:
        raise ValueError("That code is not valid.")
    codes = _new_recovery_codes()

    def mutate(policy: dict[str, Any]) -> None:
        policy["users"][name]["totp"]["recovery"] = [_recovery_tag(c) for c in codes]

    _update(path, mutate)
    _audit("recovery_codes_regenerated", name, {"username": name})
    return codes


def disable_totp(
    username: str,
    code: str | None = None,
    path: str | None = None,
    actor: str | None = None,
    verified: bool = False,
) -> None:
    """Turn two-factor sign-in off (`verified`: see regenerate_recovery_codes).

    The account turns it off for itself with a current code (or a recovery code). Anyone else is reset by an
    owner only, which also signs the account out everywhere; it is how a lost device is recovered from.
    """
    name = _normalize_username(username)
    if actor is None:
        raise PermissionError("Turning off two-factor sign-in needs a signed-in account.")
    who = _normalize_username(actor)
    if who == name:
        if not totp_enabled(name, path):
            raise ValueError("Two-factor sign-in is not on for this account.")
        if not verified and (not code or not _consume_second_factor(name, code, path)[0]):
            raise ValueError("Turning off two-factor sign-in needs a valid authentication or recovery code.")
    elif not is_owner(who, path):
        raise PermissionError("Only an owner can reset another account's two-factor sign-in.")

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        if not user.get("totp") and not user.get("totp_pending"):
            raise ValueError("Two-factor sign-in is not on for this account.")
        user.pop("totp", None)
        user.pop("totp_pending", None)
        if who != name:
            user["token_version"] = int(user.get("token_version", 0)) + 1

    _update(path, mutate)
    _audit("totp_disabled" if who == name else "totp_reset", who, {"username": name})


def prompt_second_factor() -> str:
    """Ask for an authentication or recovery code on a terminal."""
    return getpass.getpass("Authentication code (or a recovery code): ")


def _normalize_workspace(workspace: str) -> str:
    ws_id = sanitize_workspace_id(workspace)
    if not ws_id:
        raise ValueError("Invalid workspace name.")
    return ws_id


def _normalize_permissions(permissions: list[str]) -> list[str]:
    """Clean a permission list. "none" and an empty list both mean no permissions."""
    cleaned = [p.strip().lower().replace("-", "_") for p in permissions if p and p.strip()]
    cleaned = [p for p in cleaned if p != "none"]
    unknown = [p for p in cleaned if p not in PERMISSIONS]
    if unknown:
        raise ValueError(f"Unknown permission(s): {', '.join(unknown)}. Choose from {', '.join(PERMISSIONS)}.")
    return sorted(set(cleaned))


def _group_permissions(policy: dict[str, Any], user: dict[str, Any], workspace_id: str) -> list[str] | None:
    """Union of the account's group grants on a workspace; any group's denial wins. None when no group applies.

    Within one group a grant on the workspace beats that group's grant on all workspaces.
    """
    groups = policy.get("groups") or {}
    found = False
    perms: set[str] = set()
    for name in user.get("groups") or []:
        grants = (groups.get(name) or {}).get("grants") or {}
        role = grants.get(workspace_id, grants.get("*"))
        if role is None:
            continue
        if role == "none":
            return []
        found = True
        perms |= set(ROLE_PERMISSIONS[role])
    return sorted(perms) if found else None


def effective_permissions(policy: dict[str, Any], user: dict[str, Any], workspace_id: str) -> list[str]:
    """Resolve what an account may do on one workspace (see the module docstring for the order)."""
    grants = user.get("grants") or {}
    if workspace_id in grants:
        return list(grants[workspace_id])
    if "default_permissions" in user:
        return list(user["default_permissions"])
    from_groups = _group_permissions(policy, user, workspace_id)
    if from_groups is not None:
        return from_groups
    if user.get("role") in ROLE_PERMISSIONS:
        return sorted(ROLE_PERMISSIONS[user["role"]])
    return default_permissions_of(policy)


def authorize(username: str, workspace_id: str, permission: str, path: str | None = None) -> bool:
    if permission not in PERMISSIONS:
        raise ValueError(f"Unknown permission: {permission}")
    if not username or not _WORKSPACE_ID.fullmatch(workspace_id or ""):
        return False
    policy = load_policy(path)
    user = policy["users"].get(username)
    if not user:
        return False
    return permission in effective_permissions(policy, user, workspace_id)


def role_of(user: dict[str, Any] | None) -> str | None:
    """The account's own role, or None when it follows the policy's default role."""
    role = (user or {}).get("role")
    return role if role in ROLE_PERMISSIONS else None


def is_admin(username: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username) if username else None
    return role_of(user) in ADMIN_ROLES


def is_owner(username: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username) if username else None
    return role_of(user) == "owner"


def has_admin(path: str | None = None) -> bool:
    """True once at least one administrator account exists."""
    return any(role_of(info) in ADMIN_ROLES for info in load_policy(path)["users"].values())


def require_admin(actor: str, path: str | None = None) -> str:
    """Raise PermissionError unless `actor` is an administrator. Returns the actor's name."""
    if not is_admin(actor, path):
        raise PermissionError(f"'{actor}' is not an administrator. Only administrators may change accounts and permissions.")
    return actor


def _authorize_change(actor: str | None, path: str | None) -> str:
    """Admin gate for every mutating call. Entry points pass the signed-in actor; None means trusted in-process code."""
    if actor is None:
        return "system"
    return require_admin(actor, path)


def _audit(event: str, actor: str, data: dict[str, Any], workspace_id: str = "") -> None:
    audit_record(event, workspace_id=workspace_id, session_id="", data={"actor": actor, **data})


def get_user_clearance(username: str, path: str | None = None) -> str:
    """Return the clearance level assigned to a user, defaulting to 'public'."""
    if not username:
        return DEFAULT_CLEARANCE
    user = load_policy(path)["users"].get(username)
    if not user:
        return DEFAULT_CLEARANCE
    clearance = str(user.get("clearance", DEFAULT_CLEARANCE)).strip().lower()
    return clearance if clearance in CLEARANCE_LEVELS else DEFAULT_CLEARANCE


def allowed_clearances(user_clearance: str) -> list[str]:
    """Return list of clearance levels a user with user_clearance is authorized to view.

    Clearances are hierarchical: secret >= confidential >= internal >= public.
    """
    clean = user_clearance.strip().lower() if user_clearance else DEFAULT_CLEARANCE
    if clean not in CLEARANCE_LEVELS:
        clean = DEFAULT_CLEARANCE
    idx = CLEARANCE_LEVELS.index(clean)
    return list(CLEARANCE_LEVELS[:idx + 1])


def clearance_covers(reviewer_clearance: str, required_clearance: str) -> bool:
    """True when a reviewer's clearance is at least the clearance of the material under review."""
    return required_clearance in allowed_clearances(reviewer_clearance)


def allowed_workspaces(username: str, workspace_ids: list[str], path: str | None = None) -> list[str]:
    """Return workspaces where the user has query, review, or auto_approve permissions."""
    return [
        ws for ws in workspace_ids
        if any(authorize(username, ws, p, path) for p in PERMISSIONS)
    ]


def _normalize_username(raw: str) -> str:
    name = raw.strip().lower()
    if not _USERNAME.fullmatch(name):
        raise ValueError("Usernames are 2-64 characters: lowercase letters, digits, '.', '-', '_'.")
    return name


def _normalize_clearance(clearance: str) -> str:
    clean = clearance.strip().lower()
    if clean not in CLEARANCE_LEVELS:
        raise ValueError(f"Invalid clearance '{clearance}'. Allowed levels: {', '.join(CLEARANCE_LEVELS)}")
    return clean


def _existing(policy: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in policy["users"]:
        raise ValueError(f"User '{name}' does not exist.")
    user: dict[str, Any] = policy["users"][name]
    return user


def _owner_count(policy: dict[str, Any]) -> int:
    return sum(1 for info in policy["users"].values() if role_of(info) == "owner")


def _normalize_role(role: str, choices: tuple[str, ...] = ROLES) -> str:
    clean = role.strip().lower()
    if clean not in choices:
        raise ValueError(f"Unknown role '{role}'. Choose from {', '.join(choices)}.")
    return clean


def _check_role_change(policy: dict[str, Any], actor: str | None, name: str, new_role: str | None) -> None:
    """Only an owner assigns or removes the owner role, and the last owner keeps it."""
    current = role_of(policy["users"][name])
    if "owner" not in (current, new_role) or current == new_role:
        return
    if actor is not None and role_of(policy["users"].get(actor)) != "owner":
        raise PermissionError("Only an owner may assign or remove the owner role.")
    if current == "owner" and _owner_count(policy) <= 1:
        raise ValueError("Cannot remove the last owner (the last administrator who can manage owners).")


def add_user(
    username: str,
    password: str,
    admin: bool = False,
    clearance: str = DEFAULT_CLEARANCE,
    path: str | None = None,
    actor: str | None = None,
    role: str | None = None,
) -> str | None:
    """Create an account and return its role.

    Without a role it follows the policy's default role. `admin=True` without a role makes the first
    administrator the owner and later ones admins.
    """
    name = _normalize_username(username)
    check_password_strength(password, name)
    clean_clearance = _normalize_clearance(clearance)
    clean_role = _normalize_role(role) if role else None
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> str | None:
        if name in policy["users"]:
            raise ValueError(
                f"User '{name}' already exists. To reset the password, run: make set-password ACCOUNT={name}"
            )
        chosen = clean_role or (("admin" if _owner_count(policy) else "owner") if admin else None)
        if chosen == "owner" and _owner_count(policy) and actor is not None \
                and role_of(policy["users"].get(actor)) != "owner":
            raise PermissionError("Only an owner may assign or remove the owner role.")
        user: dict[str, Any] = {
            "password_hash": hash_password(password),
            "clearance": clean_clearance,
            "grants": {},
            "uid": secrets.token_hex(8),
            "token_version": 0,
        }
        if chosen:
            user["role"] = chosen
        policy["users"][name] = user
        return chosen

    chosen = _update(path, mutate)
    _audit("user_created", who, {"username": name, "role": chosen or "default", "clearance": clean_clearance})
    return chosen


def set_password(username: str, password: str, path: str | None = None, actor: str | None = None) -> None:
    """Set or reset an account's password. Grants and admin status are kept; open sessions are revoked."""
    name = _normalize_username(username)
    check_password_strength(password, name)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        user["password_hash"] = hash_password(password)
        user["token_version"] = int(user.get("token_version", 0)) + 1

    _update(path, mutate)
    _audit("password_changed", who, {"username": name})


def revoke_sessions(username: str, path: str | None = None, actor: str | None = None) -> None:
    """Invalidate every session token issued to an account."""
    name = _normalize_username(username)
    if actor is not None and _normalize_username(actor) == name:
        who = name
    else:
        who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        user["token_version"] = int(user.get("token_version", 0)) + 1

    _update(path, mutate)
    _audit("sessions_revoked", who, {"username": name})


def set_user_clearance(username: str, clearance: str, path: str | None = None, actor: str | None = None) -> None:
    """Assign a clearance level to an existing user account."""
    name = _normalize_username(username)
    clean_clearance = _normalize_clearance(clearance)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> str:
        user = _existing(policy, name)
        previous = str(user.get("clearance", DEFAULT_CLEARANCE))
        user["clearance"] = clean_clearance
        return previous

    previous = _update(path, mutate)
    _audit("clearance_changed", who, {"username": name, "from": previous, "to": clean_clearance})


def set_role(username: str, role: str | None, path: str | None = None, actor: str | None = None) -> None:
    """Set an account's role, or clear it (None or "default") so it follows the policy's default role."""
    name = _normalize_username(username)
    clean = None if role is None or role.strip().lower() == "default" else _normalize_role(role)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> str | None:
        user = _existing(policy, name)
        _check_role_change(policy, actor, name, clean)
        previous = role_of(user)
        if clean:
            user["role"] = clean
        else:
            user.pop("role", None)
        return previous

    previous = _update(path, mutate)
    _audit("role_changed", who, {"username": name, "from": previous or "default", "to": clean or "default"})


def set_admin(username: str, admin: bool, path: str | None = None, actor: str | None = None) -> None:
    """Make an account an administrator (role "admin") or remove that (it then follows the default role)."""
    name = _normalize_username(username)
    current = role_of(load_policy(path)["users"].get(name))
    if admin and current in ADMIN_ROLES:
        _authorize_change(actor, path)
        return
    if not admin and current not in ADMIN_ROLES:
        _authorize_change(actor, path)
        return
    owners = _owner_count(load_policy(path))
    set_role(name, ("admin" if owners else "owner") if admin else None, path=path, actor=actor)


def set_default_role(role: str, path: str | None = None, actor: str | None = None) -> str:
    """Set the role every account without a role of its own holds on every workspace."""
    clean = _normalize_role(role, WORKSPACE_ROLES)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> dict[str, Any]:
        previous = dict(policy.get("defaults") or {})
        policy["defaults"] = {"role": clean}
        return previous

    previous = _update(path, mutate)
    _audit("defaults_changed", who, {"from": previous, "to": {"role": clean}})
    return clean


def set_default_permissions(permissions: list[str], path: str | None = None, actor: str | None = None) -> list[str]:
    """Set the default as a permission list. It is stored as the matching role, or as a custom set."""
    perms = _normalize_permissions(permissions)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> dict[str, Any]:
        previous = dict(policy.get("defaults") or {})
        policy["defaults"] = _canonical_defaults({"permissions": perms})
        return previous

    previous = _update(path, mutate)
    _audit("defaults_changed", who, {"from": previous, "to": {"permissions": perms}})
    return perms


def get_default_permissions(path: str | None = None) -> list[str]:
    return default_permissions_of(load_policy(path))


def get_default_role(path: str | None = None) -> str | None:
    """The default role, or None when the default is a custom permission set."""
    return (load_policy(path).get("defaults") or {}).get("role")


@dataclass
class ReviewReport:
    """How answers get released, and the gaps that would leave an answer waiting with nobody to decide.

    "open" mode: the global defaults include auto_approve, so every account releases answers that pass
    every automated check without a person reading them (the out-of-the-box single-user setup).
    "team" mode: auto_approve is not a default; answers wait for a reviewer unless an account holds
    auto_approve through its own grants.
    """

    mode: str
    defaults: list[str]
    accounts: int
    warnings: list[str] = field(default_factory=list)
    reviewers: dict[str, list[str]] = field(default_factory=dict)
    auto_approve_holdouts: list[str] = field(default_factory=list)
    # The subset of warnings where a paused answer would wait with nobody able to decide it.
    gaps: list[str] = field(default_factory=list)


def review_mode_report(workspace_ids: list[str], path: str | None = None) -> ReviewReport:
    """Describe the release mode and every workspace where a paused answer could not be decided.

    A paused answer needs a reviewer who holds review on the workspace, is not the requester, and is
    cleared at least as high as the requester (the same rules human_in_the_loop_node enforces).
    """
    policy = load_policy(path)
    users = policy["users"]
    defaults = default_permissions_of(policy)
    report = ReviewReport(
        mode="open" if "auto_approve" in defaults else "team",
        defaults=defaults,
        accounts=len(users),
    )

    def clearance(name: str) -> str:
        level = str(users[name].get("clearance", DEFAULT_CLEARANCE)).strip().lower()
        return level if level in CLEARANCE_LEVELS else DEFAULT_CLEARANCE

    without_reviewer: list[str] = []
    uncovered: dict[str, set[str]] = {}
    holdouts: set[str] = set()
    for ws in workspace_ids:
        perms = {name: effective_permissions(policy, info, ws) for name, info in users.items()}
        requesters = sorted(n for n, p in perms.items() if "query" in p)
        reviewers = sorted(n for n, p in perms.items() if "review" in p)
        report.reviewers[ws] = reviewers
        if report.mode == "team":
            holdouts |= {n for n, p in perms.items() if "auto_approve" in p}
        if not requesters:
            continue
        if not reviewers:
            without_reviewer.append(ws)
            continue
        for requester in requesters:
            if not any(r != requester and clearance_covers(clearance(r), clearance(requester)) for r in reviewers):
                uncovered.setdefault(ws, set()).add(requester)

    if report.mode == "open" and report.accounts > 1:
        report.warnings.append(
            f"open review mode with {report.accounts} accounts: answers that pass every automated check are "
            "released without a person reading them. For team use run 'make team-mode ADMIN=<admin>'."
        )
        classified = _workspaces_with_classified_documents(workspace_ids)
        if classified:
            report.warnings.append(
                f"confidential or secret documents are indexed in: {', '.join(classified)}. In open mode their "
                "answers are released to cleared accounts without review."
            )
    if without_reviewer:
        report.gaps.append(
            f"no reviewer on: {', '.join(without_reviewer)}. Answers that fail an automated check, or come from "
            "accounts without auto_approve, wait there until someone holds review "
            "('make set-role ACCOUNT=<u> ROLE=reviewer ADMIN=<admin>')."
        )
    for ws, names in sorted(uncovered.items()):
        report.gaps.append(
            f"workspace '{ws}': no reviewer other than the requester is cleared high enough to release answers "
            f"for {', '.join(sorted(names))}."
        )
    report.warnings.extend(report.gaps)
    report.auto_approve_holdouts = sorted(holdouts)
    return report


def _workspaces_with_classified_documents(workspace_ids: list[str]) -> list[str]:
    from rag_lab.retrieval.sparse_search import indexed_clearances

    classified = {"confidential", "secret"}
    return [ws for ws in workspace_ids if indexed_clearances(ws, base_dir=settings.bm25_dir) & classified]


def enable_team_mode(workspace_ids: list[str], path: str | None = None, actor: str | None = None) -> ReviewReport:
    """Stop releasing answers without review by default. Only auto_approve is removed from the defaults.

    Refuses while no account holds review anywhere: every paused answer would wait with nobody able to decide.
    """
    before = review_mode_report(workspace_ids, path)
    if not any(before.reviewers.values()):
        raise ValueError(
            "No account holds review on any workspace, so every paused answer would wait with nobody able to "
            "decide. Appoint a reviewer first: make grant ACCOUNT=<u> WS=all PERMS=query,review ADMIN=<admin>"
        )
    if before.mode == "open" and get_default_role(path) == "trusted":
        set_default_role("member", path=path, actor=actor)
    elif before.mode == "open":
        set_default_permissions([p for p in before.defaults if p != "auto_approve"], path=path, actor=actor)
    else:
        _authorize_change(actor, path)
    return review_mode_report(workspace_ids, path)


TEAM_MODE_SUGGESTION_USERS = 3
GROUPS_SUGGESTION_USERS = 6


def growth_suggestions(workspace_ids: list[str], path: str | None = None) -> list[str]:
    """Next steps as the team grows. Suggestions only: nothing here changes the policy."""
    policy = load_policy(path)
    users = policy["users"]
    defaults = policy.get("defaults") or {}
    suggestions: list[str] = []
    if len(users) > TEAM_MODE_SUGGESTION_USERS and "auto_approve" in default_permissions_of(policy):
        suggestions.append(
            f"{len(users)} accounts and a default role that releases answers without review "
            f"({defaults.get('role', 'custom')}): consider 'make team-mode ADMIN=<admin>' "
            "after appointing a reviewer."
        )
    if len(users) >= GROUPS_SUGGESTION_USERS and not policy.get("groups"):
        suggestions.append(
            f"{len(users)} accounts and no groups: consider groups to grant roles per team "
            "('make create-group GROUP=<g> ADMIN=<admin>')."
        )
    owners = [n for n, info in users.items() if role_of(info) == "owner"]
    if len(users) > TEAM_MODE_SUGGESTION_USERS and len(owners) == 1:
        suggestions.append(
            f"one owner ({owners[0]}): consider a second owner so the policy is not locked to one person "
            "('make set-role ACCOUNT=<u> ROLE=owner ADMIN=<owner>')."
        )
    if "permissions" in defaults:
        suggestions.append(
            f"the default is a custom permission set ({'/'.join(defaults['permissions']) or 'none'}), not a role: "
            "consider 'make set-default ROLE=<member|trusted|reviewer|none> ADMIN=<admin>'."
        )
    return suggestions


def describe_review_mode(report: ReviewReport) -> list[str]:
    """Human-readable lines for CLIs and the status check."""
    if report.mode == "open":
        lines = ["Review mode: open (auto_approve is a default; answers that pass every check release without review)"]
    else:
        lines = ["Review mode: team (answers wait for a reviewer unless the account holds auto_approve itself)"]
    if report.auto_approve_holdouts:
        lines.append(f"Accounts still on auto_approve through their own grants: {', '.join(report.auto_approve_holdouts)}")
    return lines


def set_grant(
    username: str,
    workspace: str,
    permissions: list[str],
    path: str | None = None,
    actor: str | None = None,
) -> None:
    """Set an account's permissions on one workspace, replacing whatever applied there.

    An empty list is an explicit denial: it overrides the defaults. Use reset_grant to fall back to them.
    """
    name = _normalize_username(username)
    ws_id = _normalize_workspace(workspace)
    perms = _normalize_permissions(permissions)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        user.setdefault("grants", {})[ws_id] = perms

    _update(path, mutate)
    _audit("grant_changed", who, {"username": name, "permissions": perms}, workspace_id=ws_id)


def reset_grant(username: str, workspace: str, path: str | None = None, actor: str | None = None) -> None:
    """Remove an account's explicit override on one workspace so the defaults apply there again."""
    name = _normalize_username(username)
    ws_id = _normalize_workspace(workspace)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        _existing(policy, name).setdefault("grants", {}).pop(ws_id, None)

    _update(path, mutate)
    _audit("grant_reset", who, {"username": name}, workspace_id=ws_id)


def set_user_permissions_everywhere(
    username: str,
    permissions: list[str],
    path: str | None = None,
    actor: str | None = None,
) -> None:
    """Set an account's permissions on every workspace, existing and future, clearing per-workspace overrides."""
    name = _normalize_username(username)
    perms = _normalize_permissions(permissions)
    who = _authorize_change(actor, path)
    existing_workspaces: list[str] = []
    if perms:
        try:
            from rag_lab.storage.workspace_manager import list_workspaces
            existing_workspaces = list_workspaces()
        except Exception:
            existing_workspaces = []

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        user["default_permissions"] = perms
        user["grants"] = {ws: list(perms) for ws in existing_workspaces} if perms else {}

    _update(path, mutate)
    _audit("grant_changed", who, {"username": name, "permissions": perms, "scope": "all workspaces"})


def clear_user_permissions_override(username: str, path: str | None = None, actor: str | None = None) -> None:
    """Drop an account's own default and all its per-workspace overrides, so the global defaults apply."""
    name = _normalize_username(username)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        user.pop("default_permissions", None)
        user["grants"] = {}

    _update(path, mutate)
    _audit("grant_reset", who, {"username": name, "scope": "all workspaces"})


# --- Groups -----------------------------------------------------------------

def _normalize_group(raw: str) -> str:
    name = raw.strip().lower()
    if not _USERNAME.fullmatch(name):
        raise ValueError("Group names are 2-64 characters: lowercase letters, digits, '.', '-', '_'.")
    return name


def _existing_group(policy: dict[str, Any], name: str) -> dict[str, Any]:
    groups = policy.setdefault("groups", {})
    if name not in groups:
        raise ValueError(f"Group '{name}' does not exist. Create it with: make create-group GROUP={name}")
    group: dict[str, Any] = groups[name]
    return group


def create_group(group: str, path: str | None = None, actor: str | None = None) -> None:
    name = _normalize_group(group)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        groups = policy.setdefault("groups", {})
        if name in groups:
            raise ValueError(f"Group '{name}' already exists.")
        groups[name] = {"grants": {}}

    _update(path, mutate)
    _audit("group_created", who, {"group": name})


def delete_group(group: str, path: str | None = None, actor: str | None = None) -> None:
    """Delete a group and remove it from every member."""
    name = _normalize_group(group)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        _existing_group(policy, name)
        del policy["groups"][name]
        for info in policy["users"].values():
            if name in (info.get("groups") or []):
                info["groups"] = [g for g in info["groups"] if g != name]

    _update(path, mutate)
    _audit("group_deleted", who, {"group": name})


def add_to_group(username: str, group: str, path: str | None = None, actor: str | None = None) -> None:
    name = _normalize_username(username)
    gname = _normalize_group(group)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        _existing_group(policy, gname)
        user["groups"] = sorted(set(user.get("groups") or []) | {gname})

    _update(path, mutate)
    _audit("group_member_added", who, {"group": gname, "username": name})


def remove_from_group(username: str, group: str, path: str | None = None, actor: str | None = None) -> None:
    name = _normalize_username(username)
    gname = _normalize_group(group)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        if gname not in (user.get("groups") or []):
            raise ValueError(f"'{name}' is not in group '{gname}'.")
        user["groups"] = [g for g in user["groups"] if g != gname]

    _update(path, mutate)
    _audit("group_member_removed", who, {"group": gname, "username": name})


def grant_group(
    group: str,
    workspace: str,
    role: str | None,
    path: str | None = None,
    actor: str | None = None,
) -> None:
    """Give a group a role on one workspace, or on all ("all", including future ones).

    Role "none" is an explicit denial that beats every other group's grant; None or "default" removes the grant.
    """
    gname = _normalize_group(group)
    ws_key = "*" if _is_all(workspace) else _normalize_workspace(workspace)
    clean = None if role is None or role.strip().lower() == "default" else _normalize_role(role, WORKSPACE_ROLES)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        grants = _existing_group(policy, gname).setdefault("grants", {})
        if clean is None:
            grants.pop(ws_key, None)
        else:
            grants[ws_key] = clean

    _update(path, mutate)
    _audit("group_grant_changed", who, {"group": gname, "role": clean or "default"},
           workspace_id="" if ws_key == "*" else ws_key)


def list_groups(path: str | None = None) -> list[dict[str, Any]]:
    policy = load_policy(path)
    members: dict[str, list[str]] = {}
    for user, info in policy["users"].items():
        for g in info.get("groups") or []:
            members.setdefault(g, []).append(user)
    return [
        {"group": name, "members": sorted(members.get(name, [])), "grants": dict(info.get("grants") or {})}
        for name, info in sorted((policy.get("groups") or {}).items())
    ]


def remove_workspace_grants(workspace_id: str, path: str | None = None) -> int:
    """Forget every account's override for a deleted workspace. Returns how many accounts changed."""

    def mutate(policy: dict[str, Any]) -> int:
        changed = 0
        for info in [*policy["users"].values(), *(policy.get("groups") or {}).values()]:
            if workspace_id in (info.get("grants") or {}):
                del info["grants"][workspace_id]
                changed += 1
        return changed

    return _update(path, mutate)


def delete_user(username: str, path: str | None = None, actor: str | None = None) -> None:
    """Permanently delete a user account and all of its workspace grants. The last administrator stays."""
    name = _normalize_username(username)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        _existing(policy, name)
        _check_role_change(policy, actor, name, None)
        del policy["users"][name]

    _update(path, mutate)
    _audit("user_deleted", who, {"username": name})


def reset_users(path: str | None = None, actor: str | None = None) -> None:
    """Permanently delete all user accounts and workspace grants from the access policy."""
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> int:
        count = len(policy["users"])
        policy["users"] = {}
        policy["groups"] = {}
        return count

    removed = _update(path, mutate)
    _audit("users_reset", who, {"removed": removed})


def list_users(path: str | None = None) -> list[dict[str, Any]]:
    policy = load_policy(path)
    return [
        {
            "username": name,
            "role": role_of(info),
            "admin": role_of(info) in ADMIN_ROLES,
            "clearance": str(info.get("clearance", DEFAULT_CLEARANCE)),
            "groups": list(info.get("groups") or []),
            "totp": bool(info.get("totp")),
            "grants": info.get("grants", {}),
            "default_permissions": info.get("default_permissions"),
        }
        for name, info in sorted(policy["users"].items())
    ]


def authenticate_admin(
    account: str,
    password: str | None = None,
    path: str | None = None,
    audit_path: str | None = None,
) -> str:
    """Sign an administrator in (prompting for the password when none is given). Returns the account name."""
    secret = password if password is not None else getpass.getpass(f"Password for admin account '{account}': ")
    result = sign_in(account, secret, path, audit_path, code_prompt=prompt_second_factor)
    if not result.ok:
        raise PermissionError(f"{result.describe()} Administrator sign-in refused.")
    if not is_admin(result.username, path):
        raise PermissionError(f"'{result.username}' is not an administrator. Only administrators may change accounts and permissions.")
    return result.username


def _prompt_role(username: str) -> str:
    """Ask for the role of a new account, on a terminal. Elsewhere it follows the default role."""
    if not sys.stdin.isatty():
        return "default"
    default = get_default_role() or "custom"
    print(f"Role for '{username.strip().lower()}':")
    for role in ("member", "trusted", "reviewer", "admin"):
        print(f"  {role:9} {', '.join(ROLE_PERMISSIONS[role])}{' + administration' if role in ADMIN_ROLES else ''}")
    answer = input(f"Role [{default} = the default role]: ").strip().lower()
    if not answer:
        return "default"
    return _normalize_role(answer)


def _prompt_new_password(username: str) -> str:
    print(f"Choose a password for '{username.strip().lower()}' (at least {MIN_PASSWORD_LENGTH} characters).")
    first = getpass.getpass("New password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise ValueError("Passwords do not match.")
    return first


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_all(value: str) -> bool:
    return value.strip().lower() in ("all", "*")


def build_parser() -> argparse.ArgumentParser:
    signer_parent = argparse.ArgumentParser(add_help=False)
    signer_parent.add_argument(
        "--as",
        dest="signer",
        default=argparse.SUPPRESS,
        metavar="ADMIN",
        help="Administrator performing the change (prompts for the password).",
    )

    parser = argparse.ArgumentParser(
        description="Manage local accounts, permissions, and defaults. Changes need an administrator sign-in (--as)."
    )
    parser.add_argument("--as", dest="signer", default=None, metavar="ADMIN",
                        help="Administrator performing the change (prompts for the password). "
                             "Not needed to create the first administrator.")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-user", parents=[signer_parent], help="Create an account (prompts for a password)")
    add.add_argument("username")
    add.add_argument("--admin", action="store_true", help="Make the new account an administrator")
    add.add_argument("--role", choices=[*ROLES, "default"], default=None,
                     help="Role of the new account (prompted for from the second account on; 'default' follows "
                          "the policy's default role)")
    add.add_argument(
        "--clearance",
        default=DEFAULT_CLEARANCE,
        choices=CLEARANCE_LEVELS,
        help=f"Data clearance level (default: {DEFAULT_CLEARANCE}). Options: {', '.join(CLEARANCE_LEVELS)}",
    )

    set_clr = sub.add_parser("set-clearance", parents=[signer_parent], help="Update an account's data clearance level")
    set_clr.add_argument("username")
    set_clr.add_argument("clearance", choices=CLEARANCE_LEVELS, help=f"Options: {', '.join(CLEARANCE_LEVELS)}")

    admin_p = sub.add_parser("set-admin", parents=[signer_parent], help="Make an account an administrator, or remove that status")
    admin_p.add_argument("username")
    admin_p.add_argument("--revoke", action="store_true", help="Remove administrator status")

    grant = sub.add_parser(
        "grant",
        parents=[signer_parent],
        help="Set permissions on workspaces (replaces what applied there). Workspace list, or 'all' for every workspace incl. future ones",
    )
    grant.add_argument("username")
    grant.add_argument("workspaces", help="Workspace name, comma-separated list, or 'all'")
    grant.add_argument("--perms", default="query",
                       help="Comma-separated: query,review,auto_approve. 'none' denies; 'default' falls back to the defaults")
    grant.add_argument("--role", choices=WORKSPACE_ROLES, default=None,
                       help="Grant a role's permissions instead of --perms")

    revoke = sub.add_parser("revoke", parents=[signer_parent], help="Deny an account on workspaces (or 'all')")
    revoke.add_argument("username")
    revoke.add_argument("workspaces", help="Workspace name, comma-separated list, or 'all'")

    default = sub.add_parser("set-default", parents=[signer_parent],
                             help="Set the default role (or permission set) of accounts without a role of their own")
    chosen = default.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--role", choices=WORKSPACE_ROLES)
    chosen.add_argument("--perms", help="Comma-separated: query,review,auto_approve, or 'none'")

    role_p = sub.add_parser("set-role", parents=[signer_parent], help="Set an account's role ('default' to follow the default role)")
    role_p.add_argument("username")
    role_p.add_argument("role", choices=[*ROLES, "default"])

    sub.add_parser("totp-enroll", parents=[signer_parent],
                   help="Turn on two-factor sign-in for the --as account itself (owners and administrators)")
    sub.add_parser("totp-recovery", parents=[signer_parent],
                   help="Replace your recovery codes (needs a current authentication code)")
    totp_off = sub.add_parser("totp-disable", parents=[signer_parent],
                              help="Turn off two-factor sign-in: your own (needs a code), or another account's (owner only)")
    totp_off.add_argument("username", nargs="?", help="Account to reset (default: the --as account)")

    sub.add_parser("groups", help="List groups, their members, and their workspace roles")
    cg = sub.add_parser("create-group", parents=[signer_parent], help="Create a group")
    cg.add_argument("group")
    dg = sub.add_parser("delete-group", parents=[signer_parent], help="Delete a group (members keep their own access)")
    dg.add_argument("group")
    atg = sub.add_parser("add-to-group", parents=[signer_parent], help="Add an account to a group")
    atg.add_argument("username")
    atg.add_argument("group")
    rfg = sub.add_parser("remove-from-group", parents=[signer_parent], help="Remove an account from a group")
    rfg.add_argument("username")
    rfg.add_argument("group")
    gg = sub.add_parser("grant-group", parents=[signer_parent],
                        help="Give a group a role on workspaces (or 'all'); 'none' denies, 'default' removes the grant")
    gg.add_argument("group")
    gg.add_argument("workspaces", help="Workspace name, comma-separated list, or 'all'")
    gg.add_argument("--role", required=True, choices=[*WORKSPACE_ROLES, "default"])

    reset = sub.add_parser("set-password", parents=[signer_parent], help="Set or reset an account's password (prompts)")
    reset.add_argument("username")

    kill = sub.add_parser("revoke-sessions", parents=[signer_parent], help="Sign an account out everywhere")
    kill.add_argument("username")

    del_user = sub.add_parser("delete-user", parents=[signer_parent], help="Delete a user account and all their workspace grants")
    del_user.add_argument("username")
    del_user.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    reset_all = sub.add_parser("reset-users", parents=[signer_parent], help="Delete all user accounts and workspace grants")
    reset_all.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    sub.add_parser("list", help="Show accounts, grants, defaults, and the review mode (never password hashes)")
    sub.add_parser(
        "team-mode",
        parents=[signer_parent],
        help="Require review by default: remove auto_approve from the defaults (needs an appointed reviewer)",
    )
    return parser


def _workspace_ids() -> list[str]:
    from rag_lab.storage.workspace_manager import list_workspaces

    return list_workspaces()


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def _resolve_actor(args: argparse.Namespace) -> str | None:
    """Sign the acting administrator in. The first administrator can be created without one."""
    if args.command in ("list", "groups"):
        return None
    if not has_admin() and args.command == "add-user" and args.admin:
        return None
    if not args.signer:
        raise PermissionError(
            "Changing accounts and permissions needs an administrator. Re-run with --as <admin account> "
            "(make targets: ADMIN=<account>). The first administrator can be created without one."
        )
    return authenticate_admin(args.signer)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        actor = _resolve_actor(args)
        if args.command == "add-user":
            role = args.role
            if role is None and not args.admin and load_policy()["users"]:
                role = _prompt_role(args.username)
            role = None if role == "default" else role
            password = _prompt_new_password(args.username)
            chosen = add_user(args.username, password, admin=args.admin, clearance=args.clearance,
                              actor=actor, role=role)
            label = chosen or f"default role: {get_default_role() or 'custom'}"
            print(f"Created user '{args.username.lower()}' ({label}) [{args.clearance}].")
            perms = effective_permissions(load_policy(), load_policy()["users"][args.username.lower()], "*")
            print(f"They hold {', '.join(perms) or 'no permissions'} on every workspace without a grant of its own.")
            for gap in review_mode_report(_workspace_ids()).gaps:
                print(f"Warning: {gap}")
        elif args.command == "totp-enroll":
            _totp_enroll(actor)
        elif args.command == "totp-recovery":
            _print_recovery_codes(regenerate_recovery_codes(actor or "", "", actor=actor, verified=True))
        elif args.command == "totp-disable":
            target = args.username or actor or ""
            own = target.strip().lower() == (actor or "")
            disable_totp(target, None, actor=actor, verified=own)  # --as just signed in with the code
            print(f"Two-factor sign-in is off for '{target.lower()}'." + ("" if own else " Their sessions were revoked."))
        elif args.command == "set-role":
            set_role(args.username, args.role, actor=actor)
            print(f"'{args.username.lower()}' now has role '{args.role}'.")
            for gap in review_mode_report(_workspace_ids()).gaps:
                print(f"Warning: {gap}")
        elif args.command == "groups":
            for g in list_groups():
                grants = ", ".join(f"{'all' if ws == '*' else ws}({r})" for ws, r in sorted(g["grants"].items()))
                print(f"{g['group']}: members {', '.join(g['members']) or '-'}; grants {grants or '-'}")
        elif args.command == "create-group":
            create_group(args.group, actor=actor)
            print(f"Created group '{args.group.lower()}'.")
        elif args.command == "delete-group":
            delete_group(args.group, actor=actor)
            print(f"Deleted group '{args.group.lower()}'.")
        elif args.command == "add-to-group":
            add_to_group(args.username, args.group, actor=actor)
            print(f"Added '{args.username.lower()}' to '{args.group.lower()}'.")
        elif args.command == "remove-from-group":
            remove_from_group(args.username, args.group, actor=actor)
            print(f"Removed '{args.username.lower()}' from '{args.group.lower()}'.")
        elif args.command == "grant-group":
            targets = ["all"] if _is_all(args.workspaces) else _split(args.workspaces)
            for ws in targets:
                grant_group(args.group, ws, args.role, actor=actor)
            print(f"Group '{args.group.lower()}' has role '{args.role}' on {', '.join(targets)}.")
            for gap in review_mode_report(_workspace_ids()).gaps:
                print(f"Warning: {gap}")
        elif args.command == "set-clearance":
            set_user_clearance(args.username, args.clearance, actor=actor)
            print(f"Clearance for '{args.username.lower()}' updated to '{args.clearance}'.")
        elif args.command == "set-admin":
            set_admin(args.username, not args.revoke, actor=actor)
            print(f"'{args.username.lower()}' is {'no longer' if args.revoke else 'now'} an administrator.")
        elif args.command == "set-password":
            set_password(args.username, _prompt_new_password(args.username), actor=actor)
            print(f"Password updated for '{args.username.lower()}'. Grants are unchanged; open sessions were signed out.")
        elif args.command == "revoke-sessions":
            revoke_sessions(args.username, actor=actor)
            print(f"All sessions for '{args.username.lower()}' were revoked.")
        elif args.command == "delete-user":
            if not args.yes and not _confirm(f"Are you sure you want to delete user '{args.username}'? [y/N]: "):
                print("Deletion cancelled.")
                return 0
            delete_user(args.username, actor=actor)
            print(f"Deleted user '{args.username.lower()}'.")
        elif args.command == "reset-users":
            if not args.yes and not _confirm("Are you sure you want to delete ALL user accounts? [y/N]: "):
                print("Reset cancelled.")
                return 0
            reset_users(actor=actor)
            print("All user accounts have been reset. Create a new administrator with: make add-user ACCOUNT=<name> MAKE_ADMIN=1")
        elif args.command == "set-default":
            if args.role:
                set_default_role(args.role, actor=actor)
            else:
                set_default_permissions(_split(args.perms), actor=actor)
            role = get_default_role()
            listed = ", ".join(get_default_permissions()) or "none"
            print(f"Default for accounts without a role: {role or 'custom set'} ({listed}).")
        elif args.command == "grant":
            _grant(args, actor)
        elif args.command == "revoke":
            _revoke(args, actor)
        elif args.command == "team-mode":
            report = enable_team_mode(_workspace_ids(), actor=actor)
            for line in describe_review_mode(report):
                print(line)
            for warning in report.warnings:
                print(f"Warning: {warning}")
        else:
            defaults = ", ".join(get_default_permissions()) or "none"
            print(f"Default role (accounts without a role, every workspace): {get_default_role() or 'custom'} ({defaults})")
            for user in list_users():
                flag = f" {user['role']}" if user["role"] else ""
                if user["groups"]:
                    flag += f" groups={','.join(user['groups'])}"
                if user["totp"]:
                    flag += " 2fa"
                grants = ", ".join(f"{ws}({'/'.join(p) or 'denied'})" for ws, p in user["grants"].items())
                own = user["default_permissions"]
                extras = []
                if own is not None:
                    extras.append(f"everywhere({'/'.join(own) or 'denied'})")
                if grants:
                    extras.append(grants)
                print(f"{user['username']}{flag} [{user['clearance']}]: {', '.join(extras) or 'defaults'}")
            report = review_mode_report(_workspace_ids())
            for line in describe_review_mode(report):
                print(line)
            for warning in report.warnings:
                print(f"Warning: {warning}")
    except (ValueError, PolicyIntegrityError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _print_recovery_codes(codes: list[str]) -> None:
    print("Recovery codes (each works once; keep them somewhere safe, they are not shown again):")
    for code in codes:
        print(f"  {code}")


def _totp_enroll(actor: str | None) -> None:
    if not actor:
        raise PermissionError("Sign in as the account to protect: --as <account> (make targets: ADMIN=<account>).")
    enrolment = begin_totp_enrollment(actor, actor=actor)
    print(f"Add this account to your authenticator app, then enter a code from it.\n  Secret: {enrolment.secret}\n  URI:    {enrolment.uri}")
    codes = confirm_totp_enrollment(actor, input("Code from the app: "), actor=actor)
    print(f"Two-factor sign-in is now on for '{actor}'.")
    _print_recovery_codes(codes)


def _grant(args: argparse.Namespace, actor: str | None) -> None:
    name = _normalize_username(args.username)
    _existing(load_policy(), name)
    raw = args.perms.strip().lower()
    if raw == "default":
        if _is_all(args.workspaces):
            clear_user_permissions_override(name, actor=actor)
            print(f"'{name}' now follows the default permissions everywhere.")
        else:
            for ws in _split(args.workspaces):
                reset_grant(name, ws, actor=actor)
            print(f"'{name}' follows the default permissions on {args.workspaces}.")
        return
    perms = list(ROLE_PERMISSIONS[args.role]) if args.role else _split(args.perms)
    if _is_all(args.workspaces):
        set_user_permissions_everywhere(name, perms, actor=actor)
        print(f"Set {', '.join(perms) or 'no permissions'} for '{name}' on every workspace, including future ones.")
        return
    targets = _split(args.workspaces)
    for ws in targets:
        set_grant(name, ws, perms, actor=actor)
    print(f"Set {', '.join(perms) or 'no permissions'} on {', '.join(targets)} for '{name}'.")


def _revoke(args: argparse.Namespace, actor: str | None) -> None:
    name = _normalize_username(args.username)
    _existing(load_policy(), name)
    if _is_all(args.workspaces):
        set_user_permissions_everywhere(name, [], actor=actor)
        print(f"'{name}' has no access to any workspace, including future ones.")
        return
    targets = _split(args.workspaces)
    for ws in targets:
        set_grant(name, ws, [], actor=actor)
    print(f"Denied '{name}' on {', '.join(targets)}.")


if __name__ == "__main__":
    sys.exit(main())
