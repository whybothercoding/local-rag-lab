#!/usr/bin/env python3
"""Local identity, sign-in, and workspace access control.

Users, scrypt password hashes, per-workspace grants, and the global default permissions live in
one JSON file (ACCESS_POLICY_PATH). The file carries an HMAC tag made with a local signing key
(ACCESS_KEY_PATH); if it is edited outside this module, every check fails closed.

Permissions resolve in three layers, most specific first:
  1. an explicit per-workspace grant for the account (an empty list is an explicit denial),
  2. the account's own default permissions (set with `grant <user> all`),
  3. the global defaults (DEFAULT_PERMISSIONS on a fresh install: "query" and "auto_approve").
So out of the box every account can chat on every workspace, current and future, without review. An
administrator narrows or widens that per account, per workspace, or globally. "review" is never a
default: reviewers are appointed. "admin" is a separate flag for user, grant, workspace, and indexing
management. Only administrators may change accounts, grants, defaults, or clearances.

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
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # advisory file locking for the policy file (POSIX)
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from audit import count_events_since
from audit import record as audit_record
from config import ACCESS_KEY_PATH, ACCESS_POLICY_PATH
from vault import load_or_create_key
from workspace_manager import sanitize_workspace_id

PERMISSIONS = ("query", "review", "auto_approve")
# What every account holds on every workspace until an administrator says otherwise.
DEFAULT_PERMISSIONS = ("query", "auto_approve")
POLICY_VERSION = 2
CLEARANCE_LEVELS = ("public", "internal", "confidential", "secret")
DEFAULT_CLEARANCE = "public"
MIN_PASSWORD_LENGTH = 12
LOCKOUT_MAX_FAILURES = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60
TOKEN_TTL_SECONDS = 8 * 60 * 60
IDLE_TIMEOUT_SECONDS = 30 * 60

_USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{1,63}")
_WORKSPACE_ID = re.compile(r"[a-z0-9_]{1,48}")
_SCRYPT_PARAMS = {"n": 2**17, "r": 8, "p": 1}
_SCRYPT_MAXMEM = 256 * 1024 * 1024
_DENYLIST_PATH = Path(__file__).resolve().parent / "common_passwords.txt"


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
    return load_or_create_key(path or ACCESS_KEY_PATH)


def _canonical_defaults(defaults: Any) -> dict[str, Any]:
    perms = defaults.get("permissions") if isinstance(defaults, dict) else None
    if not isinstance(perms, list) or any(p not in PERMISSIONS for p in perms):
        raise PolicyIntegrityError("Access policy defaults are malformed. Access is refused until it is restored.")
    return {"permissions": sorted(set(perms))}


def _policy_mac(users: dict[str, Any], defaults: dict[str, Any] | None = None) -> str:
    """HMAC over the users, and over the defaults when the policy has them (policy version 2)."""
    if defaults is None:
        body = json.dumps(users, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(_signing_key(), b"policy:" + body, hashlib.sha256).hexdigest()
    body = json.dumps({"users": users, "defaults": defaults}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_signing_key(), b"policy2:" + body, hashlib.sha256).hexdigest()


def new_policy() -> dict[str, Any]:
    """The policy of a fresh install: no accounts, and the out-of-the-box default permissions."""
    return {"version": POLICY_VERSION, "users": {}, "defaults": {"permissions": sorted(DEFAULT_PERMISSIONS)}}


def load_policy(path: str | None = None) -> dict[str, Any]:
    target = Path(path or ACCESS_POLICY_PATH)
    if not target.is_file():
        return new_policy()
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
    users = policy["users"]
    defaults = policy.get("defaults")
    if defaults is not None:
        defaults = _canonical_defaults(defaults)
    if not hmac.compare_digest(str(policy.get("integrity", "")), _policy_mac(users, defaults)):
        raise PolicyIntegrityError(
            f"Access policy {target} failed its integrity check. Access is refused until it is restored."
        )
    if defaults is None:
        # A policy written before defaults existed keeps its strict meaning: no access beyond explicit
        # grants, until an administrator runs `set-default`. Upgrading never widens anyone's access.
        policy["defaults"] = {"permissions": []}
    return policy


def save_policy(policy: dict[str, Any], path: str | None = None) -> None:
    target = Path(path or ACCESS_POLICY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    defaults = policy.get("defaults")
    if defaults is not None:
        defaults = _canonical_defaults(defaults)
        policy = {**policy, "defaults": defaults, "version": POLICY_VERSION}
    policy = {**policy, "integrity": _policy_mac(policy.get("users", {}), defaults)}
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".access-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


@contextlib.contextmanager
def _policy_lock(path: str | None = None):
    """Serialize read-modify-write cycles on the policy file across threads and processes."""
    target = Path(path or ACCESS_POLICY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    lock_path = target.with_name(target.name + ".lock")
    with open(lock_path, "a") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _update(path: str | None, mutate):
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
            _LAST_POLICY_BY_UID[str(user["uid"])] = str(Path(path or ACCESS_POLICY_PATH).resolve())
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
) -> SignInResult:
    """Check a password, apply the failed-attempt lockout, and issue a session token on success.

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

    if needs_rehash(load_policy(path)["users"][name]["password_hash"]):
        def upgrade(policy: dict[str, Any]) -> None:
            policy["users"][name]["password_hash"] = hash_password(password)

        _update(path, upgrade)

    audit_record("sign_in_succeeded", workspace_id="", session_id="", data={"username": name},
                 subject=name, path=audit_path)
    return SignInResult(ok=True, username=name, token=issue_token(name, path=path))


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


def effective_permissions(policy: dict[str, Any], user: dict[str, Any], workspace_id: str) -> list[str]:
    """Resolve what an account may do on one workspace: explicit grant, then account default, then global default."""
    grants = user.get("grants") or {}
    if workspace_id in grants:
        return list(grants[workspace_id])
    if "default_permissions" in user:
        return list(user["default_permissions"])
    return list((policy.get("defaults") or {}).get("permissions", []))


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


def is_admin(username: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username) if username else None
    return bool(user and user.get("admin"))


def has_admin(path: str | None = None) -> bool:
    """True once at least one administrator account exists."""
    return any(info.get("admin") for info in load_policy(path)["users"].values())


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
    return policy["users"][name]


def _admin_count(policy: dict[str, Any]) -> int:
    return sum(1 for info in policy["users"].values() if info.get("admin"))


def add_user(
    username: str,
    password: str,
    admin: bool = False,
    clearance: str = DEFAULT_CLEARANCE,
    path: str | None = None,
    actor: str | None = None,
) -> None:
    """Create an account. It starts with the global default permissions on every workspace."""
    name = _normalize_username(username)
    check_password_strength(password, name)
    clean_clearance = _normalize_clearance(clearance)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        if name in policy["users"]:
            raise ValueError(
                f"User '{name}' already exists. To reset the password, run: make set-password ACCOUNT={name}"
            )
        policy["users"][name] = {
            "password_hash": hash_password(password),
            "admin": bool(admin),
            "clearance": clean_clearance,
            "grants": {},
            "uid": secrets.token_hex(8),
            "token_version": 0,
        }

    _update(path, mutate)
    _audit("user_created", who, {"username": name, "admin": bool(admin), "clearance": clean_clearance})


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


def set_admin(username: str, admin: bool, path: str | None = None, actor: str | None = None) -> None:
    """Grant or remove administrator status. The last administrator cannot be removed."""
    name = _normalize_username(username)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> None:
        user = _existing(policy, name)
        if user.get("admin") and not admin and _admin_count(policy) <= 1:
            raise ValueError("Cannot remove the last administrator.")
        user["admin"] = bool(admin)

    _update(path, mutate)
    _audit("admin_changed", who, {"username": name, "admin": bool(admin)})


def set_default_permissions(permissions: list[str], path: str | None = None, actor: str | None = None) -> list[str]:
    """Set the global default permissions every account holds on every workspace (e.g. query + auto_approve)."""
    perms = _normalize_permissions(permissions)
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> list[str]:
        previous = list((policy.get("defaults") or {}).get("permissions", []))
        policy["defaults"] = {"permissions": perms}
        return previous

    previous = _update(path, mutate)
    _audit("defaults_changed", who, {"from": previous, "to": perms})
    return perms


def get_default_permissions(path: str | None = None) -> list[str]:
    return list(load_policy(path)["defaults"]["permissions"])


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


def review_mode_report(workspace_ids: list[str], path: str | None = None) -> ReviewReport:
    """Describe the release mode and every workspace where a paused answer could not be decided.

    A paused answer needs a reviewer who holds review on the workspace, is not the requester, and is
    cleared at least as high as the requester (the same rules human_in_the_loop_node enforces).
    """
    policy = load_policy(path)
    users = policy["users"]
    defaults = list((policy.get("defaults") or {}).get("permissions", []))
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
        report.warnings.append(
            f"no reviewer on: {', '.join(without_reviewer)}. Answers that fail an automated check, or come from "
            "accounts without auto_approve, wait there until someone holds review "
            "('make grant ACCOUNT=<u> WS=<ws> PERMS=query,review ADMIN=<admin>')."
        )
    for ws, names in sorted(uncovered.items()):
        report.warnings.append(
            f"workspace '{ws}': no reviewer other than the requester is cleared high enough to release answers "
            f"for {', '.join(sorted(names))}."
        )
    report.auto_approve_holdouts = sorted(holdouts)
    return report


def _workspaces_with_classified_documents(workspace_ids: list[str]) -> list[str]:
    import workspace_manager
    from sparse_search import indexed_clearances

    classified = {"confidential", "secret"}
    return [ws for ws in workspace_ids if indexed_clearances(ws, base_dir=workspace_manager.BM25_DIR) & classified]


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
    if before.mode == "open":
        set_default_permissions([p for p in before.defaults if p != "auto_approve"], path=path, actor=actor)
    else:
        _authorize_change(actor, path)
    return review_mode_report(workspace_ids, path)


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
            from workspace_manager import list_workspaces
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


def remove_workspace_grants(workspace_id: str, path: str | None = None) -> int:
    """Forget every account's override for a deleted workspace. Returns how many accounts changed."""

    def mutate(policy: dict[str, Any]) -> int:
        changed = 0
        for info in policy["users"].values():
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
        user = _existing(policy, name)
        if user.get("admin") and _admin_count(policy) <= 1:
            raise ValueError("Cannot delete the last administrator.")
        del policy["users"][name]

    _update(path, mutate)
    _audit("user_deleted", who, {"username": name})


def reset_users(path: str | None = None, actor: str | None = None) -> None:
    """Permanently delete all user accounts and workspace grants from the access policy."""
    who = _authorize_change(actor, path)

    def mutate(policy: dict[str, Any]) -> int:
        count = len(policy["users"])
        policy["users"] = {}
        return count

    removed = _update(path, mutate)
    _audit("users_reset", who, {"removed": removed})


def list_users(path: str | None = None) -> list[dict[str, Any]]:
    policy = load_policy(path)
    return [
        {
            "username": name,
            "admin": bool(info.get("admin")),
            "clearance": str(info.get("clearance", DEFAULT_CLEARANCE)),
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
    result = sign_in(account, secret, path, audit_path)
    if not result.ok:
        raise PermissionError(f"{result.describe()} Administrator sign-in refused.")
    if not is_admin(result.username, path):
        raise PermissionError(f"'{result.username}' is not an administrator. Only administrators may change accounts and permissions.")
    return result.username


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

    revoke = sub.add_parser("revoke", parents=[signer_parent], help="Deny an account on workspaces (or 'all')")
    revoke.add_argument("username")
    revoke.add_argument("workspaces", help="Workspace name, comma-separated list, or 'all'")

    default = sub.add_parser("set-default", parents=[signer_parent], help="Set the permissions every account holds by default")
    default.add_argument("--perms", required=True, help="Comma-separated: query,review,auto_approve, or 'none'")

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
    from workspace_manager import list_workspaces

    return list_workspaces()


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def _resolve_actor(args: argparse.Namespace) -> str | None:
    """Sign the acting administrator in. The first administrator can be created without one."""
    if args.command == "list":
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
            add_user(
                args.username,
                _prompt_new_password(args.username),
                admin=args.admin,
                clearance=args.clearance,
                actor=actor,
            )
            print(f"Created user '{args.username.lower()}'{' (admin)' if args.admin else ''} [{args.clearance}].")
            print(f"They start with the default permissions ({', '.join(get_default_permissions()) or 'none'}) on every workspace.")
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
            perms = set_default_permissions(_split(args.perms), actor=actor)
            print(f"Default permissions for every account are now: {', '.join(perms) or 'none'}.")
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
            print(f"Defaults (every account, every workspace): {defaults}")
            for user in list_users():
                flag = " admin" if user["admin"] else ""
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
    perms = _split(args.perms)
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
