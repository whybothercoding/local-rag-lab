#!/usr/bin/env python3
"""Local identity and workspace access control.

Users, scrypt password hashes, and per-workspace grants live in one JSON file
(ACCESS_POLICY_PATH). Authorization is default-deny: an account may act on a
workspace only through an explicit grant. The file is written with owner-only permissions.

Grants are per workspace: "query" (ask questions) and "review" (approve or reject answers).
"admin" is a separate flag for workspace lifecycle (create, delete, upload, sync).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import ACCESS_POLICY_PATH
from workspace_manager import sanitize_workspace_id

PERMISSIONS = ("query", "review")
MIN_PASSWORD_LENGTH = 12
_USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{1,63}")
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(digest_hex) // 2,
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


@lru_cache(maxsize=1)
def _timing_equalizer() -> str:
    """A real hash checked against unknown usernames, so login timing does not reveal which accounts exist."""
    return hash_password("timing-equalizer-not-a-password")


def load_policy(path: str | None = None) -> dict[str, Any]:
    target = Path(path or ACCESS_POLICY_PATH)
    if not target.is_file():
        return {"version": 1, "users": {}}
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def save_policy(policy: dict[str, Any], path: str | None = None) -> None:
    target = Path(path or ACCESS_POLICY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".access-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def authenticate(username: str, password: str, path: str | None = None) -> bool:
    user = load_policy(path)["users"].get(username.strip().lower())
    stored = user.get("password_hash", "") if user else ""
    if not stored:
        verify_password(password, _timing_equalizer())
        return False
    return verify_password(password, stored)


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
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Passwords must be at least {MIN_PASSWORD_LENGTH} characters.")
    policy = load_policy(path)
    if name in policy["users"]:
        raise ValueError(f"User '{name}' already exists.")
    policy["users"][name] = {"password_hash": hash_password(password), "admin": admin, "grants": {}}
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


def _prompt_new_password() -> str:
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

    sub.add_parser("list", help="Show accounts and grants (never password hashes)")

    args = parser.parse_args(argv)
    try:
        if args.command == "add-user":
            add_user(args.username, _prompt_new_password(), admin=args.admin)
            print(f"Created user '{args.username.lower()}'{' (admin)' if args.admin else ''}.")
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
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
