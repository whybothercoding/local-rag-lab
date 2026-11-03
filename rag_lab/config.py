"""Runtime configuration for the local RAG lab: one mutable `settings` object, read at call time.

Values come from the environment, with a `.env` file in the project root filling gaps (existing
environment variables win). `settings` is built once, when this module is first imported, so a caller
that redirects stores through the environment (eval, the test setup) must do so before any project
import. Code reads `settings.<field>` when it runs, never a copy taken at import, so tests and tools
can override a field with `monkeypatch.setattr(settings, "<field>", value)`.

Defaults match docker-compose.yml and the models a fresh `make setup` machine downloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv

# Importing the package sets the offline defaults (HF_HUB_OFFLINE, ANONYMIZED_TELEMETRY) and ROOT.
from rag_lab import ROOT

load_dotenv(ROOT / ".env")

# Fixed values: not configuration, so they stay plain constants.
DEFAULT_SQLITE_PATH = str(ROOT / "data" / "checkpoints.db")
DEFAULT_DB_URI = f"sqlite:///{DEFAULT_SQLITE_PATH}"
POSTGRES_DB_URI = "postgresql://rag_user:rag_password@localhost:5432/rag_checkpoints"
DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LOCAL_LLM_URL = "http://localhost:1234/v1"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- storage -------------------------------------------------------------------------------
    db_uri: str
    # PostgreSQL connections the long-lived UI checkpointer keeps open, shared by every browser session.
    checkpoint_pool_size: int
    chroma_path: str
    bm25_dir: str
    uploads_dir: str
    knowledge_base_dir: str
    audit_db_path: str
    cache_db_path: str
    access_policy_path: str
    access_key_path: str
    # Encrypts audit payloads and checkpoint state. Keep its backup custody separate from the backups themselves.
    data_key_path: str
    anchor_path: str
    backup_dir: str
    # Audit records and idle checkpoint threads older than this are deleted. Six months.
    retention_days: int
    # Keep source files bounded so one huge upload cannot exhaust memory.
    max_source_bytes: int
    # Keyword-index tokenizer (sparse_search.TOKENIZERS). Changing it re-indexes on the next sync.
    sparse_tokenizer: str

    # --- models ----------------------------------------------------------------------------------
    llm_provider: str
    ollama_base_url: str
    local_llm_url: str
    chat_model: str
    embed_model: str
    # A hung model server must not hang the app: every chat request gives up after this many seconds.
    ollama_timeout_sec: float
    embedding_provider: str
    fastembed_model: str
    reranker_model: str
    enable_reranker: bool
    models_dir: str

    # --- retrieval and grading -------------------------------------------------------------------
    semantic_cache_similarity_threshold: float
    # Relevance gate: retrieval abstains unless the best chunk clears one of these. Calibrated on the
    # eval calibration split (see eval/README.md): answerable questions scored similarity 0.72-0.91 and
    # coverage 0.50-1.00; out-of-scope ones 0.42-0.54 and 0.00-0.25. The thresholds sit in those gaps.
    relevance_min_similarity: float
    relevance_min_coverage: float
    # Self-RAG graders: fast lexical screening first, ONNX cross-encoder fallback for paraphrases.
    groundedness_min_ratio: float
    groundedness_cross_encoder_threshold: float
    relevance_cross_encoder_threshold: float
    # Retry budgets, counted separately so one loop cannot spend the other's attempts. Retrieval retries
    # rewrite the query and search again; generation retries regenerate with the grader's reason added.
    max_retrieval_retries: int
    max_generation_retries: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_uri=_env("DB_URI", DEFAULT_DB_URI),
            checkpoint_pool_size=int(_env("CHECKPOINT_POOL_SIZE", "5")),
            chroma_path=_env("CHROMA_PATH", str(ROOT / "chroma_data")),
            bm25_dir=_env("BM25_DIR", str(ROOT / "bm25_indices")),
            uploads_dir=_env("UPLOADS_DIR", str(ROOT / "uploads")),
            knowledge_base_dir=_env("KNOWLEDGE_BASE_DIR", str(ROOT / "knowledge_base")),
            audit_db_path=_env("AUDIT_DB_PATH", str(ROOT / "data" / "audit.db")),
            cache_db_path=_env("CACHE_DB_PATH", str(ROOT / "data" / "cache.db")),
            access_policy_path=_env("ACCESS_POLICY_PATH", str(ROOT / "access_policy.json")),
            access_key_path=_env("ACCESS_KEY_PATH", str(ROOT / "data" / "access.key")),
            data_key_path=_env("DATA_KEY_PATH", str(ROOT / "data" / "data.key")),
            anchor_path=_env("ANCHOR_PATH", str(ROOT / "data" / "audit_anchors.jsonl")),
            backup_dir=_env("BACKUP_DIR", str(ROOT / "backups")),
            retention_days=int(_env("RETENTION_DAYS", "183")),
            max_source_bytes=int(_env("MAX_SOURCE_BYTES", str(25 * 1024 * 1024))),
            sparse_tokenizer=_env("SPARSE_TOKENIZER", "trigram"),
            llm_provider=_env("LLM_PROVIDER", "auto"),
            ollama_base_url=_env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            local_llm_url=_env("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL),
            chat_model=_env("OLLAMA_MODEL", DEFAULT_CHAT_MODEL),
            embed_model=_env("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            ollama_timeout_sec=float(_env("OLLAMA_TIMEOUT_SEC", "120")),
            embedding_provider=_env("EMBEDDING_PROVIDER", "fastembed"),
            fastembed_model=_env("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            reranker_model=_env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
            enable_reranker=_flag("ENABLE_RERANKER", "true"),
            models_dir=_env("MODELS_DIR", str(ROOT / "models")),
            semantic_cache_similarity_threshold=float(_env("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.92")),
            relevance_min_similarity=float(_env("RELEVANCE_MIN_SIMILARITY", "0.65")),
            relevance_min_coverage=float(_env("RELEVANCE_MIN_COVERAGE", "0.40")),
            groundedness_min_ratio=float(_env("GROUNDEDNESS_MIN_RATIO", "0.25")),
            groundedness_cross_encoder_threshold=float(_env("GROUNDEDNESS_CROSS_ENCODER_THRESHOLD", "-3.5")),
            relevance_cross_encoder_threshold=float(_env("RELEVANCE_CROSS_ENCODER_THRESHOLD", "-6.0")),
            max_retrieval_retries=int(_env("MAX_RETRIEVAL_RETRIES", "2")),
            max_generation_retries=int(_env("MAX_GENERATION_RETRIES", "1")),
        )


settings = Settings.from_env()

# Former module constant name -> settings field, for tooling and the migration test.
LEGACY_NAMES = {f.name.upper(): f.name for f in fields(Settings)} | {"OLLAMA_MODEL": "chat_model", "CHAT_MODEL": "chat_model"}
