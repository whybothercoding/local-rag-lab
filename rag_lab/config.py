"""Runtime configuration loaded once for the local RAG lab.

Defaults match docker-compose.yml and the models that are expected on a
fresh `make setup` machine. A `.env` file in the project root overrides
them when present. Existing environment variables win over the file.
"""
import os

from dotenv import load_dotenv

# Importing the package sets the offline defaults (HF_HUB_OFFLINE, ANONYMIZED_TELEMETRY) and ROOT.
from rag_lab import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_SQLITE_PATH = str(ROOT / "data" / "checkpoints.db")
DEFAULT_DB_URI = f"sqlite:///{DEFAULT_SQLITE_PATH}"
POSTGRES_DB_URI = "postgresql://rag_user:rag_password@localhost:5432/rag_checkpoints"

DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LOCAL_LLM_URL = "http://localhost:1234/v1"

DB_URI = os.getenv("DB_URI", DEFAULT_DB_URI)
# PostgreSQL connections the long-lived UI checkpointer keeps open, shared by every browser session.
CHECKPOINT_POOL_SIZE = int(os.getenv("CHECKPOINT_POOL_SIZE", "5"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
OLLAMA_MODEL = CHAT_MODEL
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
# A hung model server must not hang the app: every chat request gives up after this many seconds.
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))

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

# Self-RAG Grader thresholds: fast lexical screening first, ONNX cross-encoder fallback for paraphrases
GROUNDEDNESS_MIN_RATIO = float(os.getenv("GROUNDEDNESS_MIN_RATIO", "0.25"))
GROUNDEDNESS_CROSS_ENCODER_THRESHOLD = float(os.getenv("GROUNDEDNESS_CROSS_ENCODER_THRESHOLD", "-3.5"))
RELEVANCE_CROSS_ENCODER_THRESHOLD = float(os.getenv("RELEVANCE_CROSS_ENCODER_THRESHOLD", "-6.0"))

# Self-RAG retry budgets, counted separately so one loop cannot spend the other's attempts.
# Retrieval retries rewrite the query and search again (no relevant documents, or an off-topic answer).
# Generation retries regenerate from the same documents with the grader's reason added to the prompt.
# Larger models that benefit from more attempts can raise them; each generation retry costs one model call.
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
MAX_GENERATION_RETRIES = int(os.getenv("MAX_GENERATION_RETRIES", "1"))

# Keep source files bounded so one huge upload cannot exhaust memory.
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_BYTES", str(25 * 1024 * 1024)))
