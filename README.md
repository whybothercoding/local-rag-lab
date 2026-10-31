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

### 5. Hybrid Retrieval (Dense Semantic + Sparse Lexical Fusion)
Combines dense vector embeddings with sparse keyword search:
* **Dense Retrieval**: Cosine similarity via [ChromaDB](file:///Users/theo.dev/Documents/misc/local-rag-lab/chroma_data) and fast local embeddings (ONNX `bge-small-en-v1.5`).
* **Sparse Retrieval**: Lexical term matching via SQLite FTS5 / BM25Okapi for exact matches, technical acronyms, and identifiers.
* **Fusion**: Merges candidate sets to ensure both broad semantic recall and high-precision keyword matching.

### 6. Retrieval Abstention & Evidence Thresholds ("Knowing When Not to Answer")
To prevent hallucinations on irrelevant context, the system enforces strict evidence thresholds:
* Cosine similarity cutoff ($\ge 0.65$) or query token keyword coverage ($\ge 40\%$).
* If retrieved chunks fail these criteria, the system cleanly **abstains** (explaining that documentation was not found) instead of fabricating answers from unrelated context.

### 7. State-Scoped Semantic Caching & Fingerprinting
* Query responses are cached in memory using a composite key: `hashlib.sha256(f"{workspace_id}:{index_fingerprint}:{query}")`.
* Re-indexing or updating documents automatically updates the fingerprint, ensuring instant cache invalidation with zero stale data served.

### 8. Dynamic Intent Classification & Routing
An intent router classifies incoming queries into **domain RAG** vs. conversational **chitchat**, bypassing vector retrieval for general interactions to conserve compute and avoid noise.

### 9. AI Safety & Input Guardrails
* **Adversarial Prompt Injection Defense**: Pre-retrieval heuristics detect and reject jailbreak attempts, delimiter manipulation, and system prompt override attempts.
* **Automated PII Redaction**: Regex-based masking for emails and telephone numbers runs prior to LLM processing and before writing entries to persistent audit logs.

### 10. Human-in-the-Loop (HITL) Compliance & Dual-Control Gates
* The state graph pauses before answer release at a compliance checkpoint node.
* **Dual-Control Enforcement**: Reviewers must authenticate to approve or reject responses, and requesters are strictly prohibited from approving their own queries.
* Workflows can be resumed via either the web interface or terminal CLI (`approve.py`).

### 11. Deterministic Source Provenance & Citations
Every answer sentence is tied back to traceable chunk keys (`[workspace_doc_chunk]`) and original source files, rendered in the UI with expandable citation cards for instant auditability.

### 12. Offline Benchmarking & LLM-as-a-Judge Evaluation Gate
An offline evaluation suite (`eval.py`, `eval_metrics.py`) benchmarks retrieval precision, keyword recall, groundedness, answer rate, and abstention accuracy against golden evaluation datasets (`eval/golden.json`), with optional LLM-as-a-judge support checks.

### 13. Multi-Tenant Knowledge Isolation
Workspaces (e.g., `accounting`, `legal`, `engineering`) are partitioned into isolated vector collections (`ws_<workspace_id>`) and independent SQLite FTS5 indices, preventing cross-tenant data leakage.

### 14. Model Weight Pinning & Graceful Degradation
* Embedding weights are verified against pinned SHA-256 hashes (`model_integrity.py`).
* If the primary LLM or vector store is unreachable, the system enters a degraded mode with transparent extractive fallbacks and visible warnings rather than failing silently.

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
* **Docker** (optional, only if using PostgreSQL checkpointer instead of default SQLite)

```bash
# 1. Create virtual environment and install dependencies
make setup

# 2. Download and verify pinned embedding models (requires network once)
make models

# 3. Verify cached model weights offline
make verify-models
```

---

### 2. User Accounts & Role-Based Access Control (RBAC)

The system denies all access by default until accounts and workspace permissions are granted. Passwords are encrypted using **scrypt** with automatic hash upgrades.

```bash
# Create an administrator account (required for indexing and sync)
python access.py add-user alice --admin

# Create standard analyst accounts
python access.py add-user bob
python access.py add-user charlie

# Grant workspace-specific permissions ('query', 'review', or both)
python access.py grant bob accounting --perms query,review
python access.py grant charlie accounting --perms query

# List accounts and assigned permissions (never displays hashes)
make users
# or: python access.py list

# Reset an account password
python access.py set-password bob
```

---

### 3. Document Ingestion & Workspace Sync

Place `.md`, `.txt`, `.pdf`, or `.docx` files in dedicated folders under `knowledge_base/<workspace>/`.

```bash
# Sync all workspace directories (requires admin credentials)
make sync ADMIN=alice

# Or ingest a single directory into a target workspace
python ingest.py --workspace accounting --dir ./knowledge_base/accounting --as alice
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

### 5. Compliance CLI Review (`approve.py`)

Pending Human-in-the-Loop review threads can also be managed directly from the terminal. Reviewers must authenticate and have the `review` permission:

```bash
# List all threads awaiting compliance approval
python approve.py --list

# Approve a paused thread
python approve.py --thread ui-accounting-uuid --action approved --approver bob --reason "Verified against policy"

# Reject a paused thread
python approve.py --thread ui-accounting-uuid --action rejected --approver bob --reason "Outdated policy cited"
```

---

### 6. Security, Auditing & Governance Operations

All system events (queries, guardrail checks, retrievals, and human decisions) are committed to a tamper-evident, append-only SQLite audit log (`data/audit.db`) with AES-256-GCM encrypted payloads.

```bash
# Verify the cryptographic HMAC audit hash chain
make audit-verify

# Inspect the 20 most recent audit events
make audit-tail

# Inspect answers rejected by human reviewers along with reasons
make audit-feedback

# Anchor the audit head hash (run periodically to detect truncation)
make audit-anchor

# Apply data retention policy (prunes audit logs and idle threads > 183 days)
make retention

# Create a consistent, checksummed snapshot backup (keys excluded)
make backup

# Verify a backup manifest and audit chain
make backup-verify DIR=backups/backup-<timestamp>
```

---

### 7. Evaluation & Test Suite

Verify system health, offline compliance, and retrieval quality:

```bash
# Run full unit and integration test suite (150+ tests)
make test

# Run test suite with all outbound sockets blocked
make test-offline

# Run offline RAG benchmark evaluation (eval/golden.json)
make eval

# Run evaluation with local Ollama LLM-as-a-judge support checks
make eval-judge
```

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
├── Makefile                 # Unified setup, test, sync, UI, and audit workflows
├── docker-compose.yml       # Optional PostgreSQL container for durable checkpoints
├── eval/                    # Golden test dataset and benchmark corpus
├── knowledge_base/          # Multi-tenant knowledge base folders
└── tests/                   # Comprehensive test suite (150+ unit and integration tests)
```

---

## License

Apache-2.0
