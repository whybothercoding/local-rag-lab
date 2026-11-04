import os
import socket

import pytest

from rag_lab.config import settings
from rag_lab.security.model_integrity import verify_model


def test_offline_guard_blocks_outbound_connections(monkeypatch):
    """In the offline run the guard is already installed; elsewhere it is installed for this test only."""
    import conftest

    if os.getenv("RAG_OFFLINE") == "1":
        assert socket.socket.connect is conftest.offline_connect_guard
    else:
        monkeypatch.setattr(socket.socket, "connect", conftest.offline_connect_guard)

    for address in (("1.1.1.1", 443), ("127.0.0.1", 11434)):
        with socket.socket() as sock:
            with pytest.raises(ConnectionError):
                sock.connect(address)
    # Local IPC stays available: a missing Unix socket fails as missing, not as blocked.
    with socket.socket(socket.AF_UNIX) as unix:
        with pytest.raises(FileNotFoundError):
            # Short path: macOS caps Unix socket paths at 104 bytes. Nothing is created.
            unix.connect(f"/tmp/rag-absent-{os.getpid()}.sock")


def test_pinned_embedding_model_is_present_and_verified():
    """Fails with a clear message on a fresh clone until `make models` has run."""
    verify_model(settings.fastembed_model, settings.models_dir)
