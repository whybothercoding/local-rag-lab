#!/usr/bin/env python3
"""Local identity, sign-in, and workspace access control.

Users, scrypt password hashes, and per-workspace grants live in one JSON file
(ACCESS_POLICY_PATH). The file carries an HMAC tag made with a local signing key
(ACCESS_KEY_PATH); if it is edited outside this module, every check fails closed.

Authorization is default-deny: an account may act on a workspace only through an
explicit grant. "query" and "review" are per-workspace grants. "admin" is a separate
flag for workspace lifecycle and indexing.

Identity crosses into the pipeline as an HMAC-signed session token, never as a bare name.
"""
from __future__ import annotations

import argparse
import base64
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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from audit import count_events_since
from audit import record as audit_record
from config import ACCESS_KEY_PATH, ACCESS_POLICY_PATH
from vault import load_or_create_key
from workspace_manager import sanitize_workspace_id

PERMISSIONS = ("query", "review")
MIN_PASSWORD_LENGTH = 12
LOCKOUT_MAX_FAILURES = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60
TOKEN_TTL_SECONDS = 8 * 60 * 60
IDLE_TIMEOUT_SECONDS = 30 * 60

_USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{1,63}")
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


def _policy_mac(users: dict[str, Any]) -> str:
    body = json.dumps(users, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_signing_key(), b"policy:" + body, hashlib.sha256).hexdigest()


def load_policy(path: str | None = None) -> dict[str, Any]:
    target = Path(path or ACCESS_POLICY_PATH)
    if not target.is_file():
        return {"version": 1, "users": {}}
    with open(target, encoding="utf-8") as handle:
        policy = json.load(handle)
    users = policy.get("users", {})
    if not hmac.compare_digest(str(policy.get("integrity", "")), _policy_mac(users)):
        raise PolicyIntegrityError(
            f"Access policy {target} failed its integrity check. Access is refused until it is restored."
        )
    return policy


def save_policy(policy: dict[str, Any], path: str | None = None) -> None:
    target = Path(path or ACCESS_POLICY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    policy = {**policy, "integrity": _policy_mac(policy.get("users", {}))}
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".access-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _token_mac(body: str) -> str:
    return _b64encode(hmac.new(_signing_key(), b"token:" + body.encode("ascii"), hashlib.sha256).digest())


def issue_token(username: str, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    payload = json.dumps({"sub": username, "iat": issued, "exp": issued + TOKEN_TTL_SECONDS}, separators=(",", ":"))
    body = _b64encode(payload.encode("utf-8"))
    return f"{body}.{_token_mac(body)}"


def verify_token(token: str, now: float | None = None) -> str:
    """Return the account name a valid, unexpired token was issued to, or "" for anything else."""
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(_token_mac(body), signature):
            return ""
        payload = json.loads(_b64decode(body))
        current = now if now is not None else time.time()
        if int(payload["exp"]) <= current:
            return ""
        return str(payload["sub"])
    except (ValueError, KeyError, TypeError, AttributeError):
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
        policy = load_policy(path)
        policy["users"][name]["password_hash"] = hash_password(password)
        save_policy(policy, path)

    audit_record("sign_in_succeeded", workspace_id="", session_id="", data={"username": name},
                 subject=name, path=audit_path)
    return SignInResult(ok=True, username=name, token=issue_token(name))


def authorize(username: str, workspace_id: str, permission: str, path: str | None = None) -> bool:
    if permission not in PERMISSIONS:
        raise ValueError(f"Unknown permission: {permission}")
    if not username:
        return False
    user = load_policy(path)["users"].get(username)
    if not user:
        return False
    return permission in user.get("grants", {}).get(workspace_id, [])


def is_admin(username: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username) if username else None
    return bool(user and user.get("admin"))


def allowed_workspaces(username: str, workspace_ids: list[str], path: str | None = None) -> list[str]:
    return [ws for ws in workspace_ids if authorize(username, ws, "query", path)]


def _normalize_username(raw: str) -> str:
    name = raw.strip().lower()
    if not _USERNAME.fullmatch(name):
        raise ValueError("Usernames are 2-64 characters: lowercase letters, digits, '.', '-', '_'.")
    return name


def add_user(username: str, password: str, admin: bool = False, path: str | None = None) -> None:
    name = _normalize_username(username)
    check_password_strength(password, name)
    policy = load_policy(path)
    if name in policy["users"]:
        raise ValueError(
            f"User '{name}' already exists. To reset the password, run: python access.py set-password {name}"
        )
    policy["users"][name] = {"password_hash": hash_password(password), "admin": admin, "grants": {}}
    save_policy(policy, path)


def set_password(username: str, password: str, path: str | None = None) -> None:
    """Set or reset an account's password. Existing grants and admin status are kept."""
    name = _normalize_username(username)
    check_password_strength(password, name)
    policy = load_policy(path)
    if name not in policy["users"]:
        raise ValueError(f"User '{name}' does not exist.")
    policy["users"][name]["password_hash"] = hash_password(password)
    save_policy(policy, path)


def set_grant(username: str, workspace: str, permissions: list[str], path: str | None = None) -> None:
    name = _normalize_username(username)
    ws_id = sanitize_workspace_id(workspace)
    if not ws_id:
        raise ValueError("Invalid workspace name.")
    unknown = [p for p in permissions if p not in PERMISSIONS]
    if unknown:
        raise ValueError(f"Unknown permission(s): {', '.join(unknown)}. Choose from {', '.join(PERMISSIONS)}.")
    policy = load_policy(path)
    if name not in policy["users"]:
        raise ValueError(f"User '{name}' does not exist.")
    grants = policy["users"][name].setdefault("grants", {})
    if permissions:
        grants[ws_id] = sorted(set(permissions))
    else:
        grants.pop(ws_id, None)
    save_policy(policy, path)


def list_users(path: str | None = None) -> list[dict[str, Any]]:
    users = load_policy(path)["users"]
    return [
        {"username": name, "admin": bool(info.get("admin")), "grants": info.get("grants", {})}
        for name, info in sorted(users.items())
    ]


def _prompt_new_password(username: str) -> str:
    print(f"Choose a password for '{username.strip().lower()}' (at least {MIN_PASSWORD_LENGTH} characters).")
    first = getpass.getpass("New password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise ValueError("Passwords do not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage local accounts and workspace grants.")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-user", help="Create an account (prompts for a password)")
    add.add_argument("username")
    add.add_argument("--admin", action="store_true", help="Allow creating, deleting, and indexing workspaces")

    grant = sub.add_parser("grant", help="Grant permissions on one workspace (replaces existing grants there)")
    grant.add_argument("username")
    grant.add_argument("workspace")
    grant.add_argument("--perms", default="query", help="Comma-separated: query,review")

    revoke = sub.add_parser("revoke", help="Remove all permissions on one workspace")
    revoke.add_argument("username")
    revoke.add_argument("workspace")

    reset = sub.add_parser("set-password", help="Set or reset an account's password (prompts)")
    reset.add_argument("username")

    sub.add_parser("list", help="Show accounts and grants (never password hashes)")

    args = parser.parse_args(argv)
    try:
        if args.command == "add-user":
            add_user(args.username, _prompt_new_password(args.username), admin=args.admin)
            print(f"Created user '{args.username.lower()}'{' (admin)' if args.admin else ''}.")
        elif args.command == "set-password":
            set_password(args.username, _prompt_new_password(args.username))
            print(f"Password updated for '{args.username.lower()}'. Existing grants are unchanged.")
        elif args.command == "grant":
            perms = [p.strip() for p in args.perms.split(",") if p.strip()]
            set_grant(args.username, args.workspace, perms)
            print(f"Granted {', '.join(perms)} on '{args.workspace}' to '{args.username.lower()}'.")
        elif args.command == "revoke":
            set_grant(args.username, args.workspace, [])
            print(f"Revoked access to '{args.workspace}' for '{args.username.lower()}'.")
        else:
            for user in list_users():
                grants = ", ".join(f"{ws}({'/'.join(p)})" for ws, p in user["grants"].items()) or "none"
                flag = " admin" if user["admin"] else ""
                print(f"{user['username']}{flag}: {grants}")
    except (ValueError, PolicyIntegrityError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
