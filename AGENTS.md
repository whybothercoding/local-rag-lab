# Agent Guidelines & Architecture Reference

Instructions and engineering standards for AI assistants and coding agents working on the **Local Enterprise RAG Lab** codebase.

---

## 1. Project Overview & Invariants

Local Enterprise RAG Lab is a fully local, air-gapped Compound AI retrieval and compliance engine.

* **No External Network**: Inference and pipelines run 100% offline (`HF_HUB_OFFLINE=1`). Network access is permitted only during initial model download (`make models`).
* **Core Stack**: Python 3.11–3.14, LangGraph, Ollama (`qwen2.5-coder:7b`), ONNX FastEmbed (`bge-small-en-v1.5`) & Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`), ChromaDB, SQLite FTS5 / BM25, SQLite / PostgreSQL (`psycopg-pool`), Streamlit, and an optional Rust PyO3 accelerator (`crates/rag_accel`).
* **Fail-Closed Security**: RBAC authorization, hierarchical document clearance (`public`, `internal`, `confidential`, `secret`), signed session tokens (`principal_token`), tamper-evident HMAC hash-chained audit logging, and AES-256-GCM state encryption at rest.

---

## 2. Essential Commands

Always use `make` targets (and `WS` for workspace arguments, never `WORKSPACE`):

```bash
# Setup & Models
make setup                  # Create venv and install Python dependencies
make models                 # Download & verify pinned ONNX models (requires network once)
make verify-models          # Verify model SHA-256 integrity offline
make rust-build             # (Optional) Build native Rust PyO3 accelerator (rag_accel)

# RBAC & User Management
make add-user ACCOUNT=<u_name> [ADMIN=1] [CLEARANCE=public|internal|confidential|secret]
make grant ACCOUNT=<u_name> WS=<ws|ws1,ws2|all> [PERMS=query,review]
make revoke ACCOUNT=<u_name> WS=<ws|ws1,ws2|all>
make set-clearance ACCOUNT=<u_name> CLEARANCE=<level>
make set-password ACCOUNT=<u_name>
make delete-user ACCOUNT=<u_name> [YES=1]
make users                  # List users, clearance levels, and workspace grants
make reset-users [CONFIRM=yes]

# Ingestion, UI & HITL Compliance Review
make sync ADMIN=<admin>     # Ingest/sync knowledge_base/ and uploads/ into indexes
make ui [ADMIN=<admin>]     # Launch Streamlit web UI
make reviews                # List paused threads awaiting Human-in-the-Loop review
make review [THREAD=<id>] [ACCOUNT=<reviewer>]
make approve THREAD=<id> [ACCOUNT=<reviewer>] [REASON="..."]
make reject THREAD=<id> [ACCOUNT=<reviewer>] [REASON="..."]

# Testing, Evaluation & Governance
make test                   # Run full test suite (pytest -v)
make test-offline           # Run test suite with outbound network sockets blocked (RAG_OFFLINE=1)
make eval                   # Run offline RAG benchmark evaluation (eval/golden.json)
make eval-judge             # Evaluation with local Ollama LLM-as-a-judge support checks
make audit-verify           # Verify append-only HMAC audit hash chain
make audit-tail             # View 20 most recent audit events
make audit-feedback         # View rejected answers and reviewer reasons
make audit-anchor           # Record externalizable audit head hash anchor
make retention              # Prune expired audit records and idle checkpoints
make backup                 # Create checksummed snapshot of audit, checkpoints, and policy
make backup-verify DIR=<backup_dir>
make factory-reset [CONFIRM=yes]
```

---

## 3. Architecture & LangGraph Pipeline

Graph execution in [`app.py`](app.py) follows a cyclic Self-RAG state machine:

```text
[START] ──> [guardrail] ──(pass)──> [cache_lookup] ──(miss)──> [router]
                 │                        │                        ├──(chitchat)──> [chitchat] ──> [END]
              (fail)                    (hit)                      └──(rag)──> [retrieve]
                 │                        │                                       │
                 ▼                        ▼                               [sanitize_context]
               [END]                    [END]                                     │
                                                                             [grade_docs] ──(rewrite)──┐
                                                                                  │ (pass)             │
                                                                             [generate] <──────────────┘
                                                                                  │
                                                                       [grade_hallucination] ──(retry)──┐
                                                                                  │ (grounded)          │
                                                                          [grade_relevance]             │
                                                                           ├──(irrelevant)──> [rewrite] ┘
                                                                           └──(relevant)──> [human_approval] ──> [END]
```

### State & Tenancy Invariants
* **State Schema**: `GraphState` ([`app.py`](app.py)) preserves `original_question` (post-PII redaction) separately from `sanitized_question` (which [`query_rewrite_node`](app.py) may expand) so semantic caching and review panels always key on the user's original question.
* **Clearance-Partitioned Cache**: `SEMANTIC_CACHE` and `SEMANTIC_CACHE_CITATIONS` key on `(workspace_id, index_fingerprint, clearance, original_question)` so lower-clearance users never hit higher-clearance cached answers.
* **Workspaces**: Sanitized slugs (`ws_<workspace_id>`). Each workspace unifies `knowledge_base/<ws>` and `uploads/<ws>` across its isolated Chroma collection and SQLite FTS5 index. Deleting a workspace prunes its grants from `access_policy.json`.
* **Human-in-the-Loop (Four-Eyes Principle)**: Review interrupt triggers before answer release. Self-approval (`requester == approver`) is strictly forbidden and enforced prior to resuming graph execution in both UI and CLI ([`approve.py`](approve.py)).
* **Graceful Degradation**: If dense search, cross-encoder reranking, Rust acceleration, or the local LLM is unavailable, fall back deterministically with labeled warnings; never fail silently.

---

## 4. Coding & Security Rules

* **Documentation Sync**: Keep `AGENTS.md` and `CLAUDE.md` identical at all times.
* **Access Control**: All entry points must validate caller identity through `principal_token` and workspace permissions (`query`, `review`, `admin`) via [`access.py`](access.py).
* **Sanitization & Redaction**: Sanitize file paths, reject path traversal, enforce zip-bomb/PDF resource limits in [`parser_worker.py`](parser_worker.py), and mask PII ([`redaction.py`](redaction.py)) before persisting logs or prompt contexts.
* **Testing**: Any bug fix or feature must include unit/integration coverage in [`tests/`](tests/) and pass `make test`.
