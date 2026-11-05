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
# Chosen by `make eval-llm` (eval/README.md): multilingual, Apache 2.0, about 5 GB in memory.
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LOCAL_LLM_URL = "http://localhost:1234/v1"


# Model profiles: each fixes the embedding model, the reranker, the prefixes the embedder expects, and the
# thresholds calibrated for those models' score scales with `make calibrate-models` on the eval
# calibration split (distributions in eval/README.md). MODEL_PROFILE picks one;
# any single variable (FASTEMBED_MODEL, RELEVANCE_MIN_SIMILARITY, ...) still overrides its field.
# Every model must be pinned in rag_lab/security/model_integrity.py; `make models` fetches the profile's.
MODEL_PROFILES: dict[str, dict[str, str]] = {
    "en": {
        "fastembed_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "embedding_query_prefix": "",
        "embedding_passage_prefix": "",
        "relevance_min_similarity": "0.65",
        "relevance_min_rerank_score": "0.0",
        "groundedness_cross_encoder_threshold": "-3.5",
        "relevance_cross_encoder_threshold": "-6.0",
        "semantic_cache_similarity_threshold": "0.92",
        "relevance_gate": "similarity",
        "rerank_keep_margin": "1000",
        "chitchat_min_similarity": "2.0",
    },
    "multilingual": {
        "fastembed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "reranker_model": "BAAI/bge-reranker-v2-m3-int8",
        "embedding_query_prefix": "",
        "embedding_passage_prefix": "",
        "relevance_min_similarity": "0.30",
        "relevance_min_rerank_score": "-4.5",
        "groundedness_cross_encoder_threshold": "-3.0",
        "relevance_cross_encoder_threshold": "-6.0",
        "semantic_cache_similarity_threshold": "0.92",
        "relevance_gate": "reranker",
        "rerank_keep_margin": "5.0",
        "chitchat_min_similarity": "0.56",
    },
    "multilingual-large": {
        "fastembed_model": "intfloat/multilingual-e5-large",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        # e5 was trained with these prefixes; without them its similarities are not meaningful.
        "embedding_query_prefix": "query: ",
        "embedding_passage_prefix": "passage: ",
        "relevance_min_similarity": "0.81",
        "relevance_min_rerank_score": "-4.5",
        "groundedness_cross_encoder_threshold": "-3.0",
        "relevance_cross_encoder_threshold": "-6.0",
        "semantic_cache_similarity_threshold": "0.94",
        "relevance_gate": "reranker",
        "rerank_keep_margin": "5.0",
        "chitchat_min_similarity": "0.88",
    },
}
DEFAULT_MODEL_PROFILE = "multilingual"


def _profile() -> dict[str, str]:
    name = os.getenv("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if name not in MODEL_PROFILES:
        raise ValueError(f"Unknown MODEL_PROFILE {name!r}. Use one of: {', '.join(MODEL_PROFILES)}.")
    return MODEL_PROFILES[name]


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
    model_profile: str
    fastembed_model: str
    reranker_model: str
    embedding_query_prefix: str
    embedding_passage_prefix: str
    enable_reranker: bool
    models_dir: str

    # --- retrieval and grading -------------------------------------------------------------------
    semantic_cache_similarity_threshold: float
    # Relevance gate: retrieval abstains unless the best chunk clears one of these. Calibrated on the
    # eval calibration split (see eval/README.md): answerable questions scored similarity 0.72-0.91 and
    # coverage 0.50-1.00; out-of-scope ones 0.42-0.54 and 0.00-0.25. The thresholds sit in those gaps.
    relevance_min_similarity: float
    # Relevance gate: "reranker" (the best passage's cross-encoder logit, used when the reranker runs) or
    # "similarity" (vector similarity or word coverage). The English profile's reranker cannot judge other
    # languages, so it keeps the similarity gate.
    relevance_gate: str
    relevance_min_rerank_score: float
    # Reranker-gated profiles keep a passage only if it clears the relevance floor and scores within this
    # many logits of the best one; the best passage is always kept.
    rerank_keep_margin: float
    # Router: a short question this similar to a greeting or thanks example is chitchat (above 1 = off).
    chitchat_min_similarity: float
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
        profile = _profile()
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
            model_profile=_env("MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
            fastembed_model=_env("FASTEMBED_MODEL", profile["fastembed_model"]),
            reranker_model=_env("RERANKER_MODEL", profile["reranker_model"]),
            embedding_query_prefix=_env("EMBEDDING_QUERY_PREFIX", profile["embedding_query_prefix"]),
            embedding_passage_prefix=_env("EMBEDDING_PASSAGE_PREFIX", profile["embedding_passage_prefix"]),
            enable_reranker=_flag("ENABLE_RERANKER", "true"),
            models_dir=_env("MODELS_DIR", str(ROOT / "models")),
            semantic_cache_similarity_threshold=float(_env("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", profile["semantic_cache_similarity_threshold"])),
            relevance_min_similarity=float(_env("RELEVANCE_MIN_SIMILARITY", profile["relevance_min_similarity"])),
            relevance_gate=_env("RELEVANCE_GATE", profile["relevance_gate"]),
            relevance_min_rerank_score=float(_env("RELEVANCE_MIN_RERANK_SCORE", profile["relevance_min_rerank_score"])),
            rerank_keep_margin=float(_env("RERANK_KEEP_MARGIN", profile["rerank_keep_margin"])),
            chitchat_min_similarity=float(_env("CHITCHAT_MIN_SIMILARITY", profile["chitchat_min_similarity"])),
            relevance_min_coverage=float(_env("RELEVANCE_MIN_COVERAGE", "0.40")),
            groundedness_min_ratio=float(_env("GROUNDEDNESS_MIN_RATIO", "0.25")),
            groundedness_cross_encoder_threshold=float(_env("GROUNDEDNESS_CROSS_ENCODER_THRESHOLD", profile["groundedness_cross_encoder_threshold"])),
            relevance_cross_encoder_threshold=float(_env("RELEVANCE_CROSS_ENCODER_THRESHOLD", profile["relevance_cross_encoder_threshold"])),
            max_retrieval_retries=int(_env("MAX_RETRIEVAL_RETRIES", "2")),
            max_generation_retries=int(_env("MAX_GENERATION_RETRIES", "1")),
        )


settings = Settings.from_env()

# Former module constant name -> settings field, for tooling and the migration test.
LEGACY_NAMES = {f.name.upper(): f.name for f in fields(Settings)} | {"OLLAMA_MODEL": "chat_model", "CHAT_MODEL": "chat_model"}
