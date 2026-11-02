# Local Enterprise RAG Lab

An enterprise-ready, fully local **Compound AI** retrieval, reasoning, and compliance engine built with **LangGraph**, **Ollama (`qwen2.5-coder:7b`)**, **ChromaDB**, **SQLite FTS5 / BM25**, and **PostgreSQL / SQLite**.

Designed for complete data sovereignty, this system operates **100% air-gapped** with zero external network requests during inference. It integrates multi-tenant knowledge workspaces, self-reflective corrective RAG loops, multi-format document ingestion, and compliance-grade Human-in-the-Loop (HITL) review gates.

---

## Implemented AI Engineering Concepts

This repository serves as a reference implementation for key AI engineering patterns and production RAG architectures:

### 1. Compound AI Systems Architecture
Rather than treating LLM calls as a monolithic black box, the system orchestrates specialized, modular components: deterministic guardrails, semantic caching, intent routers, hybrid retrievers, document & hallucination graders, response generators, and compliance checkpoints.

### 2. Cyclic Agentic State Machines (LangGraph)
Uses **LangGraph** to model the RAG pipeline as a directed state graph (`GraphState`) featuring conditional branching, state cycles (retries and query rewriting loops), and durable execution interruptions.

### 3. Self-Reflective RAG (Self-RAG)
Implements dynamic self-assessment loops to verify output quality before answers are finalized:
* **Document Relevance Grading (`grade_docs`)**: Evaluates whether retrieved chunks contain sufficient relevant information to answer the question.
* **Hallucination Detection (`grade_hallucination`)**: Validates that generated statements are factually grounded in the retrieved citations.
* **Answer Relevance Grading (`grade_relevance`)**: Assesses whether the final answer directly addresses the original user prompt.

### 4. Corrective RAG (CRAG) & Adaptive Query Rewriting
When retrieved documents fail relevance grading or an answer is found irrelevant, the pipeline autonomously reformulates the user query (`rewrite`) and retries retrieval. An integrated **circuit breaker** caps retries at 2 to prevent infinite cycles.

### 5. Hybrid Retrieval & Cross-Encoder Reranking
Combines dense vector embeddings, sparse lexical search, and neural cross-encoder reranking:
* **Dense Retrieval**: Cosine similarity via ChromaDB and fast local embeddings (ONNX `bge-small-en-v1.5`).
* **Sparse Retrieval**: Lexical term matching via SQLite FTS5 / BM25 for exact matches, technical acronyms, and identifiers.
* **Reciprocal Rank Fusion (RRF)**: Merges candidate sets into an initial top candidate list.
* **Neural Cross-Encoder Reranker (`reranker.py`)**: Shortlisted candidates undergo deep cross-attention re-scoring with a local, pinned ONNX model (`Xenova/ms-marco-MiniLM-L-6-v2`) to ensure that chunks directly answering the query are prioritized above partial keyword distractions.

### 6. Retrieval Abstention & Evidence Thresholds ("Knowing When Not to Answer")
To prevent hallucinations on irrelevant context, the system enforces strict evidence thresholds:
* Cosine similarity cutoff ($\ge 0.65$) or query token keyword coverage ($\ge 40\%$).
* If retrieved chunks fail these criteria, the system cleanly **abstains** (explaining that documentation was not found) instead of fabricating answers from unrelated context.

### 7. State-Scoped & Clearance-Partitioned Semantic Caching
* Approved query responses and their citations are cached in memory using a composite key: `hashlib.sha256(f"{workspace_id}:{clearance}:{index_fingerprint}:{original_question}")`.
* Partitioning by user clearance ensures a lower-clearance user never hits a cached answer generated from higher-clearance documents, while keying on `original_question` ensures queries rewritten during CRAG loops still hit the cache on subsequent asks.
* Re-indexing or updating documents automatically rotates `index_fingerprint`, guaranteeing instant cache invalidation with zero stale data served.

### 8. Dynamic Intent Classification & Routing
An intent router classifies incoming queries into **domain RAG** vs. conversational **chitchat**, bypassing vector retrieval for general interactions to conserve compute and avoid noise.

### 9. AI Safety, Prompt Injection Defense & Outbound DLP
* **Direct & Indirect Prompt Injection Defense**: Pre-retrieval heuristics detect direct user jailbreaks (`guardrail`). Post-retrieval context scanning (`sanitize_context`) strips untrusted documents containing role spoofing, delimiter attacks, instruction overrides, and exfiltration beacons prior to generator evaluation.
* **Real-Time Outbound Data Loss Prevention (DLP)**: Generated LLM responses undergo real-time redaction for emails, phone numbers, and sensitive identifiers before release. Audit logs are cryptographically hash-chained and PII-masked.

### 10. Human-in-the-Loop (HITL) Compliance & Dual-Control Gates
* The state graph pauses before answer release at a compliance checkpoint node.
* **Dual-Control Enforcement**: Reviewers must authenticate to approve or reject responses, and requesters are strictly prohibited from approving their own queries.
* Workflows can be resumed via either the web interface or terminal CLI (`approve.py`).

### 11. Deterministic Source Provenance & Citations
Every answer sentence is tied back to traceable chunk keys (`[workspace_doc_chunk]`) and original source files, rendered in the UI with expandable citation cards for instant auditability.

### 12. Offline Benchmarking & LLM-as-a-Judge Evaluation Gate
An offline evaluation suite (`eval.py`, `eval_metrics.py`) benchmarks retrieval precision, keyword recall, groundedness, answer rate, and abstention accuracy against golden evaluation datasets (`eval/golden.json`), with optional LLM-as-a-judge support checks.

### 13. Multi-Tenant Knowledge Isolation & Document-Level RBAC
* **Workspace Isolation**: Workspaces (e.g., `accounting`, `legal`, `engineering`) are partitioned into isolated vector collections (`ws_<workspace_id>`) and independent SQLite FTS5 indices, preventing cross-tenant data leakage.
* **Document-Level Clearance Filtering**: Documents declare hierarchical clearance tags (`public`, `internal`, `confidential`, `secret`) in frontmatter or headers (unrecognized tags fail closed during ingestion). Vector search (`ChromaDB where` + post-query guard) and keyword search (`SQLite FTS5 WHERE clearance IN (...)`) strictly enforce the caller's authorized clearance, rendering classified documents completely invisible to unauthorized users.

### 14. Model Weight Pinning, Optional Rust Acceleration & Graceful Degradation
* Embedding and cross-encoder reranker weights are verified against pinned SHA-256 hashes (`model_integrity.py`).
* **Optional Native Rust Accelerator (`crates/rag_accel`)**: Compiles via PyO3/Maturin (`make rust-build`) to accelerate indirect prompt-injection scanning (`RegexSet` single-pass automaton) and document chunking (`split_text`) with 100% parity and automatic zero-config fallback to pure Python.
* If the primary LLM, cross-encoder, or vector store is unreachable, the system enters a degraded mode with transparent extractive fallbacks and visible warnings rather than failing silently.

---

## Architecture Flow

```
[START]
   │
[guardrail] ──(fail)──> [END] (Adversarial Prompt Violation)
   │ (pass)
[cache_lookup] ──(hit)──> [END] (Direct Semantic Cache Hit)
   │ (miss)
[router] ──(chitchat)──> [chitchat] ──> [END]
   │ (rag)
[retrieve] (Hybrid ChromaDB Dense + SQLite FTS5 / BM25 Sparse)
   │
[sanitize_context] (Zero Trust Indirect Prompt Injection Defense)
   │
[grade_docs] ──(rewrite)──> [rewrite] ──> [retrieve]
   │ (approved / max retries)
[generate]
   │
[grade_hallucination] ──(retry)──> [generate]
   │ (grounded / circuit break)
[grade_relevance] ──(irrelevant)──> [rewrite]
   │ (relevant)
[human_approval] (HITL Compliance Checkpoint)
   │
 [END]
```

---

## Usage Instructions

### 1. Prerequisites & Environment Setup

* **Python**: `>= 3.11` (tested up to 3.14)
* **Ollama** (optional, recommended for local generative LLM): `ollama pull qwen2.5-coder:7b`
* **Rust Toolchain** (optional, for compiling the native `rag_accel` PyO3 extension)
* **Docker** (optional, only if using PostgreSQL checkpointer instead of default SQLite)

```bash
# 0. View all available make commands
make help

# 1. Create virtual environment and install dependencies
make setup

# 2. Download and verify pinned embedding & reranker models (requires network once)
make models

# 3. Verify cached model weights offline
make verify-models

# 4. (Optional) Compile native Rust accelerator for high-throughput context scanning & chunking
make rust-build

# 5. Inspect full system status (models, accelerator, LLM, RBAC, workspaces, audit health)
make status
```

---

### 2. User Accounts & Role-Based Access Control (RBAC)

The system denies all access by default until accounts and workspace permissions are granted. Passwords are encrypted using **scrypt** with automatic hash upgrades.

All operations can be managed directly via `make` commands without activating the virtual environment:

```bash
# Create an administrator account (required for indexing and sync)
make add-user ACCOUNT=alice ADMIN=1

# Create standard analyst accounts with clearance levels (public, internal, confidential, secret)
make add-user ACCOUNT=bob CLEARANCE=internal
make add-user ACCOUNT=charlie CLEARANCE=public

# Update a user's clearance level
make set-clearance ACCOUNT=bob CLEARANCE=confidential

# Grant workspace-specific permissions ('query', 'review', or both)
# Accepts a single workspace, comma-separated list, or 'all'
make grant ACCOUNT=bob WS=accounting PERMS=query,review
make grant ACCOUNT=bob WS=accounting,strategy PERMS=query,review
make grant ACCOUNT=charlie WS=all PERMS=query

# Revoke access to workspaces (accepts a single workspace, comma-separated list, or 'all')
make revoke ACCOUNT=charlie WS=accounting
make revoke ACCOUNT=bob WS=all

# List accounts, clearance, and assigned permissions (never displays hashes)
make users

# Reset an account password (prompts securely)
make set-password ACCOUNT=bob

# Delete an individual user and all their grants
make delete-user ACCOUNT=bob
# Or bypass confirmation prompt:
make delete-user ACCOUNT=bob YES=1

# Delete ALL user accounts and grants
make reset-users
# Or bypass confirmation prompt:
make reset-users CONFIRM=yes
```

---

### 3. Workspace Management & Document Ingestion

Place `.md`, `.txt`, `.pdf`, `.docx`, or other supported files in dedicated folders under `knowledge_base/<workspace>/` (or manage workspaces directly via `make`):

```bash
# List all workspaces with file counts, chunk counts, and FTS5 index status
make workspaces

# Create or delete a workspace from the CLI
make create-workspace WS=legal
make delete-workspace WS=legal YES=1

# Sync all workspace directories (requires admin credentials)
make sync ADMIN=alice

# Sync a single workspace (auto-discovers knowledge_base/<ws>/ or accepts a custom DIR, plus optional FORCE=1)
make sync ADMIN=alice WS=accounting
make sync ADMIN=alice WS=accounting DIR=./knowledge_base/accounting FORCE=1
```

* Workspaces are automatically sanitized (e.g., `Accounting & Tax` becomes `accounting_tax`).
* Generates Chroma collections (`ws_<workspace_id>`) and SQLite FTS5 keyword indexes (`bm25_indices/<workspace_id>.db`).
* Idempotent: unchanged files are automatically skipped based on SHA-256 manifests.

---

### 4. Interactive Web Application

Launch the Streamlit web interface:

```bash
make ui ADMIN=alice
# Or launch without re-indexing:
make ui
```

Open `http://localhost:8501` in your browser:
1. **Sign In**: Authenticate with your username and password to obtain an HMAC-signed session token.
2. **Select Workspace**: Switch between permitted workspaces in the sidebar.
3. **Ask Questions**: Submit natural language queries.
4. **Inspect Citations**: Expand citation cards beneath responses to verify exact chunk origins and document text.
5. **Human Review Drawer**: If an answer requires compliance sign-off, authorized reviewers can approve or reject the output directly in the UI.

---

### 5. Compliance Review (Human-in-the-Loop)

Pending Human-in-the-Loop review threads can be managed directly via `make` commands. Under Separation of Duties, a query must be approved by an authorized reviewer with `review` permissions other than the requester:

```bash
# List all threads awaiting compliance approval
make reviews

# Interactively review a paused thread (prompts to approve/reject and records decisions)
make review
# Or specify a thread and reviewer:
make review THREAD=ui-accounting-uuid ACCOUNT=bob

# Directly approve a paused thread
make approve THREAD=ui-accounting-uuid ACCOUNT=bob REASON="Verified against policy"

# Directly reject a paused thread
make reject THREAD=ui-accounting-uuid ACCOUNT=bob REASON="Outdated policy cited"
```

---

### 6. Security, Auditing & Governance Operations

All system events (queries, guardrail checks, retrievals, and human decisions) are committed to a tamper-evident, append-only SQLite audit log (`data/audit.db`) with AES-256-GCM encrypted payloads.

```bash
# Verify the cryptographic HMAC audit hash chain
make audit-verify

# Inspect recent audit events (optional N=<count> and WS=<workspace>)
make audit-tail
make audit-tail N=50 WS=accounting

# Inspect answers rejected by human reviewers along with reasons
make audit-feedback
make audit-feedback N=20 WS=accounting

# Anchor the audit head hash (run periodically to detect truncation)
make audit-anchor

# Export decrypted audit trail as JSONL (to stdout or OUT=<file>)
make audit-export OUT=audit-export.jsonl

# Apply data retention policy (prunes audit logs and idle threads > 183 days by default)
make retention
make retention DAYS=90

# Create, list, and verify consistent checksummed snapshot backups (keys excluded)
make backup
make backups
make backup-verify DIR=backups/backup-<timestamp>

# Zero-trace factory reset (wipes all users, audit logs, checkpoints, keys, and vectors)
make factory-reset
# Or bypass confirmation prompt:
make factory-reset CONFIRM=yes
```

---

### 7. Evaluation & Test Suite

Verify system health, offline compliance, and retrieval quality:

```bash
# Run full unit and integration test suite
make test

# Run test suite with all outbound sockets blocked
make test-offline

# Run offline RAG benchmark evaluation (eval/golden.json)
make eval

# Run evaluation with local Ollama LLM-as-a-judge support checks
make eval-judge

# Print retrieval similarity and keyword coverage signals on the calibration split
make eval-calibrate

# Export LangGraph architecture diagram to docs/architecture.mmd
make graph
```

---

### 8. Storage Architecture & Docker Evaluation

The engine supports two storage topologies for persisting LangGraph conversation state and checkpoints:

#### Default Mode: Embedded SQLite (Zero-Ops & Self-Contained)
* **Status:** Default out-of-the-box (`data/checkpoints.db`).
* **Operational Footprint:** **Zero Docker required.** Checkpoints, audit trails (`data/audit.db`), and FTS5 keyword indices run natively as embedded SQLite databases with Write-Ahead Logging (WAL) and AES-256-GCM encryption-at-rest.

#### Optional Mode: PostgreSQL via Docker Compose (`make up`)
The repository includes a `docker-compose.yml` to run PostgreSQL (`postgres:16-alpine`) as an external checkpointer (`DB_URI=postgresql://...`).

**Is Dockerized PostgreSQL "Enterprise-Ready"?**
* **For Small Teams (1 to ~30 people / Internal Air-Gapped Labs):**
  - **Verdict:**  **Practical & Recommended.**
  - Provides instant, reproducible database provisioning without cloud infrastructure overhead or cloud egress costs.
  - State is persisted to a Docker volume (`pgdata`), easily backed up via standard host snapshots or `pg_dump` jobs.
* **For Large Enterprise Teams (50+ to Thousands of users):**
  - **Verdict:** ⚠️ **Single-container Docker is NOT enterprise-ready.**
  - A standalone container on a single host lacks **High Availability (HA)**, automatic multi-AZ failover, read replica scaling, and point-in-time recovery (PITR).
  - **Recommended Enterprise Alternative:** Large enterprises should point `DB_URI` to a **Cloud-Managed Service** (AWS Aurora/RDS, GCP Cloud SQL, Azure Database for PostgreSQL) or deploy PostgreSQL via a **Kubernetes Database Operator** (e.g. [CloudNativePG](https://cloudnative-pg.io/) or [Crunchy Data PGO](https://access.crunchydata.com/documentation/postgres-operator/latest/)) with continuous WAL archiving (`pgBackRest` / `wal-g`).

---

## Repository Structure

```text
local-rag-lab/
├── app.py                   # LangGraph state machine, nodes, Self-RAG conditional routes
├── workspace_manager.py     # Multi-tenant workspace isolation and collection lifecycle
├── ingest.py                # Multi-format document parser (.md, .txt, .pdf, .docx) & indexer
├── parsers.py               # Document parsing implementations (pypdf, docx, html/txt)
├── parser_worker.py         # Sandboxed child process parser with time/memory caps
├── sparse_search.py         # SQLite FTS5 / BM25Okapi sparse lexical search engine
├── reranker.py              # Local ONNX cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
├── embedding_provider.py    # Local ONNX FastEmbed embedding engine
├── llm_provider.py          # Ollama and local OpenAI-compatible chat model connectors
├── model_integrity.py       # SHA-256 model weight pinning and offline integrity checks
├── access.py                # RBAC user accounts, scrypt hashing, signed session tokens
├── approve.py               # CLI tool for inspecting and resolving HITL review threads
├── audit.py                 # Append-only, hash-chained tamper-evident audit logger
├── vault.py                 # AES-256-GCM data sealing and state encryption at rest
├── retention.py             # Audit log and idle checkpoint thread retention manager
├── backup.py                # Checksummed backup creation and verification utility
├── redaction.py             # Automated PII detection and regex masking
├── ui.py                    # Streamlit web application with citation drawers & review UI
├── eval.py                  # Offline evaluation benchmark runner
├── eval_metrics.py          # RAG evaluation metrics (precision, recall, groundedness)
├── crates/rag_accel/        # Optional native Rust PyO3 accelerator (context scanning & chunking)
├── Makefile                 # Unified setup, test, sync, UI, review, and audit workflows
├── docker-compose.yml       # Optional PostgreSQL container for durable checkpoints
├── eval/                    # Golden test dataset and benchmark corpus
├── knowledge_base/          # Multi-tenant knowledge base folders
└── tests/                   # Comprehensive test suite (190+ unit and integration tests)
```

---

## License

Apache-2.0
