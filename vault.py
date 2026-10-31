"""Encryption at rest and keyed tags for the local data stores.

Audit payloads and checkpoint state are sealed with AES-256-GCM under a data key kept in its own
owner-only file (DATA_KEY_PATH). Every sealed audit payload is bound to its row metadata through
additional authenticated data, so a ciphertext moved to another row fails to open.

Values written before sealing existed are read as plaintext, so existing databases still load.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from config import DATA_KEY_PATH

__all__ = [
    "InvalidTag",
    "EncryptedSerde",
    "anchor_mac",
    "chain_mac",
    "load_or_create_key",
    "open_bytes",
    "open_text",
    "seal_bytes",
    "seal_text",
    "subject_tag",
]

TEXT_PREFIX = "enc1:"
_BYTES_PREFIX = b"ENC1"
_NONCE_BYTES = 12


def load_or_create_key(path: str | None = None) -> bytes:
    """Read a 32-byte key, creating it with owner-only permissions on first use."""
    target = Path(path or DATA_KEY_PATH)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(32))
    return target.read_bytes()


def _aead() -> AESGCM:
    return AESGCM(load_or_create_key())


def seal_bytes(plaintext: bytes, aad: bytes) -> bytes:
    nonce = secrets.token_bytes(_NONCE_BYTES)
    return _BYTES_PREFIX + nonce + _aead().encrypt(nonce, plaintext, aad)


def open_bytes(blob: bytes, aad: bytes) -> bytes:
    """Decrypt a sealed value. Unsealed values are returned unchanged (legacy plaintext)."""
    if not blob.startswith(_BYTES_PREFIX):
        return blob
    nonce = blob[len(_BYTES_PREFIX):len(_BYTES_PREFIX) + _NONCE_BYTES]
    return _aead().decrypt(nonce, blob[len(_BYTES_PREFIX) + _NONCE_BYTES:], aad)


def seal_text(text: str, aad: bytes) -> str:
    return TEXT_PREFIX + base64.urlsafe_b64encode(seal_bytes(text.encode("utf-8"), aad)).decode("ascii")


def open_text(stored: str, aad: bytes) -> str:
    if not stored.startswith(TEXT_PREFIX):
        return stored
    raw = base64.urlsafe_b64decode(stored[len(TEXT_PREFIX):])
    return open_bytes(raw, aad).decode("utf-8")


def subject_tag(value: str) -> str:
    """A keyed, non-reversible tag for an identifier such as an account name. Lets us count by subject without storing it."""
    return hmac.new(load_or_create_key(), b"subject:" + value.encode("utf-8"), hashlib.sha256).hexdigest()


def anchor_mac(body: str) -> str:
    return hmac.new(load_or_create_key(), b"anchor:" + body.encode("utf-8"), hashlib.sha256).hexdigest()


def chain_mac(body: str) -> str:
    """Keyed hash for the audit chain. Without the data key, an edited log cannot be re-chained to look intact."""
    return hmac.new(load_or_create_key(), b"chain:" + body.encode("utf-8"), hashlib.sha256).hexdigest()


class EncryptedSerde(JsonPlusSerializer):
    """LangGraph serializer that seals checkpoint state. Legacy unsealed checkpoints still load."""

    def dumps_typed(self, obj):
        kind, data = super().dumps_typed(obj)
        return kind, seal_bytes(data, b"checkpoint")

    def loads_typed(self, data):
        kind, blob = data
        return super().loads_typed((kind, open_bytes(blob, b"checkpoint")))
