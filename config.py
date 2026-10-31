"""Runtime configuration loaded once for the local RAG lab.

Defaults match docker-compose.yml and the models that are expected on a
fresh `make setup` machine. A `.env` file in the project root overrides
them when present. Existing environment variables win over the file.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEFAULT_SQLITE_PATH = str(ROOT / "data" / "checkpoints.db")
DEFAULT_DB_URI = f"sqlite:///{DEFAULT_SQLITE_PATH}"
POSTGRES_DB_URI = "postgresql://rag_user:rag_password@localhost:5432/rag_checkpoints"

DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LOCAL_LLM_URL = "http://localhost:1234/v1"

DB_URI = os.getenv("DB_URI", DEFAULT_DB_URI)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
OLLAMA_MODEL = CHAT_MODEL
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed")
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODELS_DIR = os.getenv("MODELS_DIR", str(ROOT / "models"))

CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_data"))
BM25_DIR = os.getenv("BM25_DIR", str(ROOT / "bm25_indices"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", str(ROOT / "uploads"))
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(ROOT / "knowledge_base"))

# Keep source files bounded so one huge upload cannot exhaust memory.
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_BYTES", str(25 * 1024 * 1024)))
