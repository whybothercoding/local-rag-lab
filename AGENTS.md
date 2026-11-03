# Agent Guidelines & Architecture Reference

Instructions and engineering standards for AI assistants and coding agents working on the **Local Enterprise RAG Lab** codebase.

---

## 1. Project Overview & Invariants

Local Enterprise RAG Lab is a fully local, air-gapped Compound AI retrieval and compliance engine.

* **No External Network**: Inference and pipelines run 100% offline (`HF_HUB_OFFLINE=1`). Network access is permitted only during initial model download (`make models`).
* **Core Stack**: Python 3.11–3.14, LangGraph, Ollama (`qwen2.5-coder:7b`), ONNX FastEmbed (`bge-small-en-v1.5`) & Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`), ChromaDB, SQLite FTS5 / BM25, SQLite / PostgreSQL (`psycopg-pool`), Streamlit, and an optional Rust PyO3 accelerator (`crates/rag_accel`).
* **Fail-Closed Security**: Policy v2 RBAC authorization with out-of-the-box `query` + `auto_approve` defaults, admin-only user/workspace management (`--as <admin>`), hierarchical document clearance (`public`, `internal`, `confidential`, `secret`), revocable signed session tokens (`principal_token` bound to `uid` + `token_version`), tamper-evident HMAC hash-chained audit logging, and thread-bound AES-256-GCM state and cache encryption at rest (`ENC2` / `SEAL1`).

---

## 2. Essential Commands

Always use `make` targets (and `WS` for workspace arguments, never `WORKSPACE`). Mutating user and workspace commands require `ADMIN=<admin>` once the first administrator is created:

```bash
# Setup, Models & Health
make help                   # Show categorized command reference (default target)
make status                 # End-to-end health check (models, accelerator, LLM, RBAC, workspaces, audit)
make setup                  # Create venv and install Python dependencies
make models                 # Download & verify pinned ONNX models (requires network once)
make verify-models          # Verify model SHA-256 integrity offline
make rust-build             # (Optional) Build native Rust PyO3 accelerator (rag_accel)

# RBAC & User Management (Admin-Gated)
make add-user ACCOUNT=<u_name> [ADMIN=1] [CLEARANCE=public|internal|confidential|secret] [AS=<admin>]
make grant ACCOUNT=<u_name> WS=<ws|ws1,ws2|all> [PERMS=query,review,auto_approve] ADMIN=<admin>
make revoke ACCOUNT=<u_name> WS=<ws|ws1,ws2|all> ADMIN=<admin>
make set-default [PERMS=query,auto_approve|none] ADMIN=<admin>
make team-mode ADMIN=<admin>  # Drop auto_approve from defaults (open -> team review mode); needs a reviewer
make set-admin ACCOUNT=<u_name> ADMIN=<admin> [REVOKE=1]
make set-clearance ACCOUNT=<u_name> CLEARANCE=<level> ADMIN=<admin>
make set-password ACCOUNT=<u_name> ADMIN=<admin>
make revoke-sessions ACCOUNT=<u_name> ADMIN=<admin>
make delete-user ACCOUNT=<u_name> ADMIN=<admin> [YES=1]
make users                  # List users, clearance levels, global defaults, and workspace grants
make reset-users ADMIN=<admin> [CONFIRM=yes]

# Workspaces, Ingestion, UI & HITL Compliance Review
make workspaces             # List workspaces with file, chunk, and FTS5 status
make create-workspace WS=<ws> ADMIN=<admin>
make delete-workspace WS=<ws> ADMIN=<admin> [YES=1]
make sync ADMIN=<admin> [WS=<ws>] [DIR=<path>] [FORCE=1]
make ui [ADMIN=<admin>]     # Launch Streamlit web UI
make reviews                # List paused threads awaiting Human-in-the-Loop review
make review [THREAD=<id>] [ACCOUNT=<reviewer>]
make approve THREAD=<id> [ACCOUNT=<reviewer>] [REASON="..."]
make reject THREAD=<id> [ACCOUNT=<reviewer>] [REASON="..."]

# Testing, Evaluation & Governance
make test                   # Run full test suite (pytest -v)
make test-offline           # Run test suite with outbound network sockets blocked (RAG_OFFLINE=1)
make lint                   # Ruff lint gate (rule set pinned in pyproject.toml)
make coverage               # Test suite with per-module coverage report
make eval                   # Run offline RAG benchmark evaluation (eval/golden.json)
make eval-judge             # Evaluation with local Ollama LLM-as-a-judge support checks
make eval-calibrate         # Print retrieval similarity and coverage signals on calibration split
make graph                  # Export LangGraph topology to docs/architecture.mmd
make audit-verify           # Verify append-only HMAC audit hash chain and anchors
make audit-tail [N=20] [WS=<ws>]
make audit-feedback [N=20] [WS=<ws>]
make audit-anchor           # Record externalizable audit head hash anchor
make audit-export [OUT=audit-export.jsonl]
make retention [DAYS=183]   # Prune expired audit records and idle checkpoints
make backup                 # Create checksummed snapshot of audit, checkpoints, and policy
make backups                # List existing backup snapshots
make backup-verify DIR=<backup_dir>
make clean                  # Clean caches and temporary test artifacts
make factory-reset [CONFIRM=yes]
```

---

## 3. Architecture & LangGraph Pipeline

Graph execution in [`app.py`](app.py) follows a cyclic Self-RAG state machine:

```text
[START] ──> [guardrail] ──(fail)──> [END]
                 │ (pass)
           [cache_lookup] ──(hit)──> [END]
                 │ (miss)
             [router] ──(chitchat)──> [chitchat] ──> [END]
                 │ (rag)
             [retrieve] <──────── [rewrite] <──────────────────┐
                 │                    ▲                        │
          [sanitize_context]          │                        │
                 │                    │                        │
            [grade_docs] ──(rewrite)──┘                        │
                 │ (approved | circuit_break)                  │
             [generate] <────────────────────┐                 │
                 │                           │ (retry)         │
        [grade_hallucination] ───────────────┘                 │
                 │ (grounded)                                  │
          [grade_relevance] ──(irrelevant)─────────────────────┘
                 │ (relevant | circuit_break)
          [human_approval] ──> [END]
```
`grade_hallucination` also exits to `human_approval` on `circuit_break` (not drawn). Rewrite edges spend `retry_count` (`MAX_RETRIEVAL_RETRIES`); the retry edge spends `generation_retry_count` (`MAX_GENERATION_RETRIES`) and carries `grader_feedback` into the prompt. The generated topology [`docs/architecture.mmd`](docs/architecture.mmd) is authoritative: run `make graph` after changing edges (a test fails when it is stale).

### State & Tenancy Invariants
* **State Schema**: `GraphState` ([`app.py`](app.py)) preserves `original_question` (post-PII redaction) separately from `sanitized_question` (which [`query_rewrite_node`](app.py) may expand) so semantic caching, relevance grading, and review panels always key on the user's original question.
* **Policy v2 3-Tier RBAC Resolution**: [`effective_permissions`](access.py) resolves workspace permissions in priority order: (1) explicit workspace grant (`user["grants"][ws]`, where `[]` is an explicit denial), (2) user-wide default (`user["default_permissions"]` set via `grant <user> all`), and (3) global policy default (`policy["defaults"]["permissions"]`, which defaults to `["auto_approve", "query"]` on Policy v2 so new users can chat uninterrupted out-of-the-box). Only authenticated administrators (`actor` / `--as <admin>`) can mutate users, grants, clearances, global defaults, or workspaces.
* **Clearance-Partitioned Sealed Cache**: `SEMANTIC_CACHE` and `SEMANTIC_CACHE_CITATIONS` ([`semantic_cache.py`](semantic_cache.py)) provide an LRU in-memory + AES-256-GCM sealed SQLite-persistent cache keyed on `(workspace_id, clearance, index_fingerprint, original_question)` with cosine-similarity semantic retrieval, numeric/negation signature verification, and `reviewed` status tracking so users without `auto_approve` only consume human-reviewed cache entries.
* **Fail-Closed Document Clearance**: Unrecognized explicit `clearance` tags in YAML frontmatter (case-insensitive) or `.clearance` folder files are rejected during ingestion ([`ingest.py`](ingest.py)). Both dense ([`app.py`](app.py)) and sparse ([`sparse_search.py`](sparse_search.py)) retrieval enforce `allowed_clearances` at query time and via fail-closed post-query guards that reject chunks missing a valid clearance attribute.
* **Workspaces**: Sanitized slugs (`ws_<workspace_id>`). Each workspace deterministically merges `knowledge_base/<ws>` and `uploads/<ws>` (with `knowledge_base` taking precedence on filename collisions) and performs an atomic Chroma collection swap (`__build` -> live) during `FORCE=1` re-syncs. Deleting a workspace requires admin privileges, escapes SQL `LIKE` wildcards, and prunes its grants from `access_policy.json`.
* **Human-in-the-Loop & Safety Escalation**: Answers release automatically when the user holds `auto_approve` **and** the generation is grounded, relevant, non-abstaining, and free of blocking degradation (`security_warning`, `dense_retrieval_unavailable`, `sparse_retrieval_unavailable`). Otherwise, execution pauses at `interrupt()` for a reviewer who holds `review` permission, is not the requester (`requester != approver`), and whose clearance dominates (`clearance_covers`) the thread's `principal_clearance`.
* **Hybrid Self-RAG Graders**: [`grade_hallucination_node`](app.py) and [`grade_answer_relevance_node`](app.py) combine lexical/numeric grounding checks (`_unsupported_numbers` rejects fabricated digits or number words) with an in-process pinned ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) via [`reranker.py`](reranker.py).
* **Rust Accelerator Parity & Graceful Degradation**: `crates/rag_accel` strips zero-width Unicode obfuscation and runs a single-pass `RegexSet` automaton with 100% verdict and reason parity with [`context_sanitizer.py`](context_sanitizer.py). If dense search, cross-encoder reranking, Rust acceleration, or the local LLM is unavailable, fall back deterministically with labeled warnings; never fail silently. Never switch embedding models automatically: indexes store their `embedding_fingerprint`, and a mismatch skips dense retrieval until re-index.

---

## 4. Coding & Security Rules

* **Documentation Sync**: Keep `AGENTS.md` and `CLAUDE.md` identical at all times.
* **Packaging & Configuration Standards**: Project builds, metadata, dependencies, console scripts, and tool configs (pytest, ruff, mypy) are centrally declared in [`pyproject.toml`](pyproject.toml) following PEP 517/518/621. Editable installs (`pip install -e . --no-build-isolation`) register standard console scripts (`local-rag`, `local-rag-sync`, `local-rag-access`, etc.).
* **Access Control**: All entry points must validate caller identity through `principal_token` and workspace permissions (`query`, `review`, `auto_approve`, `admin`) via [`access.py`](access.py).
* **Sanitization & Redaction**: Normalize and scan both user queries and document chunks for prompt injection ([`context_sanitizer.py`](context_sanitizer.py)), sanitize file paths, enforce zip-bomb/PDF resource limits in [`parser_worker.py`](parser_worker.py), and mask PII (emails, phones, SSNs, Luhn payment cards, mod-97 IBANs in [`redaction.py`](redaction.py)) before persisting logs or prompt contexts.
* **Testing**: Any bug fix or feature must include unit/integration coverage in [`tests/`](tests/) and pass `make test`. PostgreSQL tests use the `postgres_uri` fixture: they run against `POSTGRES_TEST_URI`, or the docker-compose server when `make up` is running (a recreated `<db>_pytest` database, also in the offline run when local), and CI fails them instead of skipping (`REQUIRE_POSTGRES=1`). The UI is tested with Streamlit `AppTest`; assert `not app.exception`.
