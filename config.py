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

# Runtime never contacts a model hub or telemetry endpoint. Only fetch_models.py may download,
# and it re-enables network access itself before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

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
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("1", "true", "yes", "on")
MODELS_DIR = os.getenv("MODELS_DIR", str(ROOT / "models"))

CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_data"))
BM25_DIR = os.getenv("BM25_DIR", str(ROOT / "bm25_indices"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", str(ROOT / "uploads"))
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(ROOT / "knowledge_base"))
AUDIT_DB_PATH = os.getenv("AUDIT_DB_PATH", str(ROOT / "data" / "audit.db"))
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", str(ROOT / "data" / "cache.db"))
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.92"))
ACCESS_POLICY_PATH = os.getenv("ACCESS_POLICY_PATH", str(ROOT / "access_policy.json"))
ACCESS_KEY_PATH = os.getenv("ACCESS_KEY_PATH", str(ROOT / "data" / "access.key"))
# Encrypts audit payloads and checkpoint state. Keep its backup custody separate from the backups themselves.
DATA_KEY_PATH = os.getenv("DATA_KEY_PATH", str(ROOT / "data" / "data.key"))
ANCHOR_PATH = os.getenv("ANCHOR_PATH", str(ROOT / "data" / "audit_anchors.jsonl"))
# Audit records and idle checkpoint threads older than this are deleted. Six months.
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "183"))
BACKUP_DIR = os.getenv("BACKUP_DIR", str(ROOT / "backups"))

# Relevance gate: retrieval abstains unless the best chunk clears one of these. Calibrated on the
# eval calibration split; see eval/README.md.
# Calibration: answerable questions scored similarity 0.72-0.91 and coverage 0.50-1.00; out-of-scope
# questions scored similarity 0.42-0.54 and coverage 0.00-0.25. The thresholds sit in those gaps.
RELEVANCE_MIN_SIMILARITY = float(os.getenv("RELEVANCE_MIN_SIMILARITY", "0.65"))
RELEVANCE_MIN_COVERAGE = float(os.getenv("RELEVANCE_MIN_COVERAGE", "0.40"))

# Keep source files bounded so one huge upload cannot exhaust memory.
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_BYTES", str(25 * 1024 * 1024)))
