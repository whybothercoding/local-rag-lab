import os
import socket

import pytest

from config import FASTEMBED_MODEL, MODELS_DIR
from model_integrity import verify_model


@pytest.mark.skipif(os.getenv("RAG_OFFLINE") != "1", reason="guard is only installed with RAG_OFFLINE=1")
def test_offline_guard_blocks_outbound_connections():
    with socket.socket() as sock:
        with pytest.raises(ConnectionError):
            sock.connect(("1.1.1.1", 443))


def test_pinned_embedding_model_is_present_and_verified():
    """Fails with a clear message on a fresh clone until `make models` has run."""
    verify_model(FASTEMBED_MODEL, MODELS_DIR)
