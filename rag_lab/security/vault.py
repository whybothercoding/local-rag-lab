"""Encryption at rest and keyed tags for the local data stores.

Audit payloads and checkpoint state are sealed with AES-256-GCM under a data key held in the OS keychain
or in its own owner-only file (DATA_KEY_PATH; see keystore.py). Every sealed audit payload is bound to its row metadata through
additional authenticated data, so a ciphertext moved to another row fails to open.

Values written before sealing existed are read as plaintext, so existing databases still load.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

import contextlib
import contextvars

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from rag_lab.security.keystore import InvalidKeyError, load_or_create_key

__all__ = [
    "InvalidTag",
    "EncryptedSerde",
    "InvalidKeyError",
    "bind_thread",
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
_BOUND_PREFIX = b"ENC2"
_NONCE_BYTES = 12

# The thread a checkpoint operation belongs to. checkpointer.py sets it around every saver call so that
# sealed state is bound to its thread and cannot be copied into another one.
_THREAD: contextvars.ContextVar[str | None] = contextvars.ContextVar("checkpoint_thread", default=None)


@contextlib.contextmanager
def bind_thread(thread_id: str):
    token = _THREAD.set(str(thread_id))
    try:
        yield
    finally:
        _THREAD.reset(token)


def _allow_legacy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _aead() -> AESGCM:
    return AESGCM(load_or_create_key())


def seal_bytes(plaintext: bytes, aad: bytes) -> bytes:
    nonce = secrets.token_bytes(_NONCE_BYTES)
    return _BYTES_PREFIX + nonce + _aead().encrypt(nonce, plaintext, aad)


def open_bytes(blob: bytes, aad: bytes, *, allow_plaintext: bool = True) -> bytes:
    """Decrypt a sealed value.

    Unsealed values are returned unchanged only when allow_plaintext is true (legacy data). Callers
    that must not accept forged plaintext pass allow_plaintext=False and get InvalidTag instead.
    """
    if not blob.startswith(_BYTES_PREFIX):
        if allow_plaintext:
            return blob
        raise InvalidTag("value is not sealed")
    nonce = blob[len(_BYTES_PREFIX):len(_BYTES_PREFIX) + _NONCE_BYTES]
    return _aead().decrypt(nonce, blob[len(_BYTES_PREFIX) + _NONCE_BYTES:], aad)


def seal_text(text: str, aad: bytes) -> str:
    return TEXT_PREFIX + base64.urlsafe_b64encode(seal_bytes(text.encode("utf-8"), aad)).decode("ascii")


def open_text(stored: str, aad: bytes, *, allow_plaintext: bool = True) -> str:
    if not stored.startswith(TEXT_PREFIX):
        if allow_plaintext:
            return stored
        raise InvalidTag("value is not sealed")
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


def _checkpoint_aad() -> bytes:
    thread = _THREAD.get()
    return b"checkpoint:" + thread.encode("utf-8") if thread is not None else b"checkpoint:unbound"


class EncryptedSerde(JsonPlusSerializer):
    """LangGraph serializer that seals checkpoint state and binds it to its thread.

    New data is written as ENC2 and can only be opened for the thread it was written for. ENC1 data
    (sealed before binding existed) still loads unless ALLOW_LEGACY_SEAL=0. Unsealed legacy data can
    be disabled strictly with allow_plaintext=False or ALLOW_LEGACY_PLAINTEXT=0.
    """

    def __init__(self, *, allow_plaintext: bool | None = None):
        super().__init__()
        self._allow_plaintext = allow_plaintext

    def dumps_typed(self, obj):
        kind, data = super().dumps_typed(obj)
        nonce = secrets.token_bytes(_NONCE_BYTES)
        sealed = _BOUND_PREFIX + nonce + _aead().encrypt(nonce, data, _checkpoint_aad())
        return kind, sealed

    def loads_typed(self, data):
        kind, blob = data
        blob = bytes(blob)
        if blob.startswith(_BOUND_PREFIX):
            nonce = blob[len(_BOUND_PREFIX):len(_BOUND_PREFIX) + _NONCE_BYTES]
            plain = _aead().decrypt(nonce, blob[len(_BOUND_PREFIX) + _NONCE_BYTES:], _checkpoint_aad())
        elif blob.startswith(_BYTES_PREFIX):
            if not _allow_legacy("ALLOW_LEGACY_SEAL", True):
                raise InvalidTag("legacy unbound checkpoint seal is disabled")
            plain = open_bytes(blob, b"checkpoint", allow_plaintext=False)
        else:
            allowed = (
                self._allow_plaintext
                if self._allow_plaintext is not None
                else _allow_legacy("ALLOW_LEGACY_PLAINTEXT", True)
            )
            if not allowed:
                raise InvalidTag("unsealed checkpoint state refused (set ALLOW_LEGACY_PLAINTEXT=1 to migrate)")
            plain = blob
        return super().loads_typed((kind, plain))
