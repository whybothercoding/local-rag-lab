# Agent Guidelines & Architecture Reference

Instructions and engineering standards for AI assistants and coding agents working on the **Local Enterprise RAG Lab** codebase.

---

## 1. Project Overview & Invariants

Local Enterprise RAG Lab is a fully local, air-gapped Compound AI retrieval and compliance engine.

* **No External Network**: Inference and pipelines must run 100% offline. Network access is permitted only during initial model download (`make models`).
* **Core Stack**: Python 3.11–3.14, LangGraph, Ollama (`qwen2.5-coder:7b`) / ONNX FastEmbed, ChromaDB, SQLite FTS5 / BM25, SQLite / PostgreSQL (`psycopg-pool`), Streamlit.
* **Fail-Closed Security**: RBAC authorization, signed session tokens (`principal_token`), tamper-evident hash-chained audit logging, AES-256-GCM state encryption at rest.

---

## 2. Essential Commands

```bash
# Setup & Models
make setup                  # Create venv and install dependencies
make models                 # Download & verify pinned embedding weights (requires network)
make verify-models          # Verify model SHA-256 integrity offline

# Testing & Evaluation
make test                   # Run full test suite (pytest -v)
make test-offline           # Run test suite with network sockets blocked (RAG_OFFLINE=1)
make eval                   # Run offline RAG benchmark evaluation (eval/golden.json)
make eval-judge             # Evaluation with Ollama LLM-as-a-judge support checks

# Ingestion & UI
make sync ADMIN=<user>      # Ingest/sync knowledge_base/ directories into indexes
make ui                     # Run Streamlit frontend application

# Security & Governance
make audit-verify           # Verify append-only HMAC-keyed audit chain
make audit-tail             # View recent audit events
make retention              # Apply audit and checkpoint data retention policy
make backup                 # Create encrypted snapshot of audit, checkpoints, and policy
```

---

## 3. Architecture & LangGraph Pipeline

Graph execution in [`app.py`](file:///Users/theo.dev/Documents/misc/local-rag-lab/app.py) follows a cyclic Self-RAG state machine:

```
[START] ──> [guardrail] ──(pass)──> [cache_lookup] ──(miss)──> [router]
                 │                        │                        ├──(chitchat)──> [chitchat] ──> [END]
              (fail)                    (hit)                      └──(rag)──> [retrieve]
                 │                        │                                       │
                 ▼                        ▼                                  [grade_docs] ──(rewrite)──┐
               [END]                    [END]                                     │ (pass)             │
                                                                             [generate] <──────────────┘
                                                                                  │
                                                                       [grade_hallucination] ──(retry)──┐
                                                                                  │ (grounded)          │
                                                                          [grade_relevance]             │
                                                                           ├──(irrelevant)──> [rewrite] ┘
                                                                           └──(relevant)──> [human_approval] ──> [END]
```

### State & Tenancy Invariants
* **State Schema**: `GraphState` ([`app.py`](file:///Users/theo.dev/Documents/misc/local-rag-lab/app.py)).
* **Workspaces**: Sanitized slugs (`ws_<workspace_id>`). Each workspace maintains its own Chroma collection and SQLite FTS5 index. Never allow cross-workspace data retrieval.
* **Human-in-the-Loop**: Review interrupt triggers before answer release. Self-approval is strictly forbidden (requester $\neq$ approver).
* **Graceful Degradation**: If dense search or LLM fails, return labeled warnings or extractive fallback; never fail silently.

---

## 4. Coding & Security Rules

* **Type Hints & Contracts**: Maintain PEP 484 type annotations and docstrings across modules.
* **Access Control**: All entry points must validate caller identity through `principal_token` and workspace permissions via [`access.py`](file:///Users/theo.dev/Documents/misc/local-rag-lab/access.py).
* **Sanitization & Redaction**: Sanitize file paths, reject path traversal, enforce zip-bomb/PDF page limits, and mask PII (email/phone) before persisting logs or prompt contexts.
* **Testing**: Any modification or new feature must include comprehensive unit/integration tests in [`tests/`](file:///Users/theo.dev/Documents/misc/local-rag-lab/tests).
