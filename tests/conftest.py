"""Shared test setup.

Set RAG_OFFLINE=1 (`make test-offline`) to block every outbound TCP connection,
including localhost. Unix domain sockets stay available for local IPC.
"""
import os
import socket

# Chroma's anonymous telemetry must never run in tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

if os.getenv("RAG_OFFLINE") == "1":
    _original_connect = socket.socket.connect

    def _guarded_connect(self, address):
        if self.family == socket.AF_UNIX:
            return _original_connect(self, address)
        raise ConnectionError(f"Network access is disabled for offline tests (attempted {address}).")

    socket.socket.connect = _guarded_connect
