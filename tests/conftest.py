"""Shared test setup.

Set RAG_OFFLINE=1 (`make test-offline`) to block every outbound TCP connection,
including localhost. Unix domain sockets stay available for local IPC.
"""
import atexit
import os
import shutil
import socket
import tempfile

# Chroma's anonymous telemetry must never run in tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Audit events from tests go to a throwaway file, never to the project's data/ directory.
_audit_dir = tempfile.mkdtemp(prefix="rag-audit-")
atexit.register(shutil.rmtree, _audit_dir, ignore_errors=True)
os.environ["AUDIT_DB_PATH"] = os.path.join(_audit_dir, "audit.db")

if os.getenv("RAG_OFFLINE") == "1":
    _original_connect = socket.socket.connect

    def _guarded_connect(self, address):
        if self.family == socket.AF_UNIX:
            return _original_connect(self, address)
        raise ConnectionError(f"Network access is disabled for offline tests (attempted {address}).")

    socket.socket.connect = _guarded_connect
