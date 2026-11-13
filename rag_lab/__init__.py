"""Local RAG Lab.

Importing the package anchors runtime data at the repository root and switches off every model-hub
and telemetry call before any submodule (chromadb, fastembed) is imported. Keep this file free of
project imports: eval.py and the test setup redirect stores through the environment first.
"""
import os
from pathlib import Path

# Data, keys, models, and indexes live at the repository root, never inside the package directory.
# RAG_LAB_HOME overrides it, e.g. for a non-editable install.
ROOT = Path(os.environ.get("RAG_LAB_HOME") or Path(__file__).resolve().parents[1]).resolve()

# Runtime never contacts a model hub or telemetry endpoint. Only fetch_models re-enables the hub,
# itself, before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
