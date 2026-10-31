# Local Enterprise RAG Lab

An enterprise-ready, locally-hosted Compound AI retrieval and compliance engine built with **LangGraph**, **Ollama (`qwen2.5-coder:7b`)**, **ChromaDB**, **BM25**, and **PostgreSQL**.

This architecture guarantees complete data sovereignty by running 100% locally with no third-party API dependencies. It incorporates advanced agentic Self-RAG loops, multi-tenant knowledge workspace isolation, multi-format document ingestion, and compliance-grade Human-in-the-Loop (HITL) review gates.

---

## Key Capabilities

- **100% Air-Gapped Local Inference**: Powered by local Ollama instances for embeddings and code/text generation.
- **Multi-Tenant Knowledge Bundles**: Create, isolate, switch, and delete knowledge domains (e.g., `accounting`, `strategy`, `legal`) with zero cross-tenant vector or index leakage.
- **Multi-Format Document Parsing**: Native extraction and recursive chunking for `.md`, `.txt`, `.pdf` (`pypdf`), and `.docx` (`python-docx`).
- **Hybrid Dense + Sparse Retrieval**: Fuses semantic vector search (ChromaDB) with keyword precision (BM25Okapi).
- **Self-RAG Corrective Evaluation**:
  - Heuristic adversarial prompt injection detection and automatic PII masking (emails, phone numbers).
  - SHA-256 state-scoped semantic query caching.
  - LLM-grounded document grading, hallucination detection, and automatic query re-writing with circuit-breaker protection (max 2 retries).
- **Durable Checkpointing & HITL Review**: LangGraph state graph saved to a PostgreSQL database via `psycopg-pool`, supporting human review interrupts via Streamlit UI or terminal CLI (`approve.py`).
- **Visual Source Provenance**: Expandable UI citation drawers showing exact document origins and chunk keys for auditability.

---

## Architecture

```
[START]
   │
[guardrail] ──(fail)──> [END] (Adversarial Prompt Violation)
   │ (pass)
[cache_lookup] ──(hit)──> [END] (Direct Semantic Cache Hit)
   │ (miss)
[router] ──(chitchat)──> [chitchat] ──> [END]
   │ (rag)
[retrieve] (Hybrid ChromaDB Dense + BM25 Sparse)
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

## Quickstart

### Prerequisites

- **Python**: `>= 3.11` (tested on 3.11–3.14)
- Nothing else for the default setup. SQLite, ONNX embeddings, and the built-in synthesizer need no services.
- Network access is needed **once**, for `make models`.

Optional backends:
- **PostgreSQL checkpoints**: `make up`, then set `DB_URI=postgresql://rag_user:rag_password@localhost:5432/rag_checkpoints`.
- **Ollama chat model**: `make ollama-pull`. The app uses it automatically when `localhost:11434` responds, and otherwise uses the built-in synthesizer.

### 1. Setup

```bash
make setup    # virtual environment and pure-Python dependencies
make models   # one-time: download the embedding model and verify its SHA-256 pins
```

After this step the application runs with no network access. `make verify-models` re-checks the cached weights offline.

### 2. Create Accounts and Grant Access

The app will not serve anyone until at least one account exists. Passwords are stored as scrypt hashes.

```bash
python access.py add-user alice --admin                 # administrator: creates and indexes workspaces
python access.py add-user bob                           # analyst
python access.py grant bob accounting --perms query,review
python access.py list
python access.py set-password bob                       # forgotten or shared password? reset it here
```

Permissions are per workspace. `query` allows asking questions. `review` allows approving or rejecting answers, but never one's own. Access is denied by default.

### 3. Verify System Health

```bash
make test          # full suite (PostgreSQL test runs when a server is reachable)
make test-offline  # same suite with every outbound connection blocked
make eval          # offline evaluation gate: retrieval, answer, and access-control metrics
```

---

## Usage

### Ingesting Documents via CLI

Put `.md`, `.txt`, `.pdf`, or `.docx` files in one folder per workspace under `knowledge_base/`, then run:

```bash
make sync ADMIN=alice   # index every folder under knowledge_base/; unchanged files are skipped
                         # indexing is an administrative action: you are asked for that account's password
```

Each folder name becomes a workspace ID (for example `Accounting & Tax` becomes `accounting_tax`). Symlinked files and folders are ignored. Each workspace gets its own ChromaDB collection (`ws_<workspace_id>`) and SQLite FTS5 keyword index (`bm25_indices/<workspace_id>.db`).

To index a single directory into a named workspace:

```bash
python ingest.py --workspace accounting --dir ./knowledge_base/accounting --as alice
```

### Answer Quality Controls

- **Abstention.** Retrieval passes evidence on only if the best chunk is similar enough to the question (cosine ≥ 0.65) or covers enough of its content words (≥ 40%). Otherwise the assistant says it could not find documentation instead of answering from unrelated text.
- **Cache invalidation.** Cached answers are keyed by workspace, query, and the workspace's index fingerprint. Re-indexing invalidates them. The cache lives in memory and is cleared on restart.
- **Visible degradation.** If the vector store, keyword index, or model is unavailable, the answer carries a warning. A model failure produces a labeled extractive answer. Nothing silently falls back.
- **Rejections as feedback.** `make audit-feedback` lists answers reviewers declined, with their reasons.
- **Local OpenAI-compatible model.** Set `LLM_PROVIDER=openai` and `LOCAL_LLM_URL` to use LM Studio, vLLM, or llama.cpp servers.
- **Measured.** `make eval` gates answer rate, retrieval, keyword recall, groundedness, abstention, and security on a held-out split. See `eval/README.md`.

### Data Governance

- **Encryption at rest.** Audit payloads and checkpoint state are sealed with AES-256-GCM (`cryptography`). Each sealed audit payload is bound to its own row, so moving it elsewhere makes it unreadable.
- **Keys.** `data/data.key` encrypts the data stores and keys the audit chain. `data/access.key` signs sessions and the access policy. Keep backup copies of both in a custody separate from the backups. Losing `data.key` makes the audit log and checkpoints unreadable, and that is intended.
- **Tamper evidence.** The audit chain is keyed, so an edited log cannot be re-hashed to look intact without the data key. Truncation of the newest records is caught by signed **anchors**: run `make audit-anchor` on a schedule and copy `data/audit_anchors.jsonl` off the host.
- **Retention: six months.** `make retention` deletes audit records and idle checkpoint threads older than 183 days. Each run is recorded in the chain, so the gap is visible. Threads awaiting human review are never deleted. Set `RETENTION_DAYS` to change the period. Nothing is archived by default. If longer retention is required, change the period rather than keeping copies.
- **Backups.** `make backup` writes a consistent copy of the audit log, anchors, checkpoints, and access policy, with a checksum manifest and no keys. `make backup-verify DIR=...` checks the copy, including the chain and anchors. Vector and keyword indexes are not backed up: rebuild them with `make sync`.
- **Administrator view.** The sidebar's audit panel verifies the chain and anchors, shows recent events, and exports the log. Each export is recorded.
- **Not covered.** Chroma and SQLite FTS5 indexes hold document text in plain form. Protect them with full-disk encryption on the host. PostgreSQL retention needs your DBA tooling.

### Security Controls

- **Signed identity.** Signing in issues an HMAC-signed session token that expires after 8 hours. The pipeline verifies the token itself. A bare account name in the input is ignored.
- **Sign-in throttling.** Five failed sign-ins in 15 minutes lock an account for that window. Failures are written to the audit log, without the password.
- **Idle timeout.** The UI signs you out after 30 minutes without activity.
- **Tamper detection.** `access_policy.json` carries an integrity tag. If it is edited outside `access.py`, access is refused until it is restored.
- **Password policy.** At least 12 characters, not on the bundled denylist (`common_passwords.txt`), and not containing the account name. Hashes use scrypt with N=2^17 and are upgraded automatically at the next sign-in.
- **Upload hygiene.** Filenames are reduced to a base name with a supported extension. Files stay inside their workspace folder, and UI uploads are capped at the source size limit.
- **Isolated parsing.** Documents are parsed in a child process with a time limit and memory cap. Office archives with zip-bomb ratios and PDFs over 1,000 pages are refused.
- **Reproducible dependencies.** `requirements.lock` pins the tested versions (Python 3.14, Linux and macOS wheels resolved). `requirements.txt` keeps the version ranges.

### Launching the Interactive Web UI

Launch the Streamlit interface:

```bash
make ui
# Or: streamlit run ui.py
```

From the web browser (`http://localhost:8501`), you can:
- **Switch Bundles**: Select an active workspace from the sidebar.
- **Create / Delete Bundles**: Manage staging folders, indices, and vector collections directly.
- **Upload Documents**: Drag and drop `.pdf`, `.docx`, `.txt`, and `.md` files to index them on the fly.

### Audit Trail

Every query, groundedness check, retrieval, and human decision is written to `data/audit.db` (override with `AUDIT_DB_PATH`).

- **Append-only and hash-chained.** Each record commits to the one before it, and `UPDATE` and `DELETE` are rejected by triggers. Editing or removing any row breaks verification from that point on.
- **Redacted by default.** Emails and phone numbers are masked before anything is written.
- **Attributed decisions.** Reviewers must enter a reviewer ID to approve or reject. An approval without one is recorded as a rejection and is not released.
- **Fail-closed.** If an event cannot be written, the answer is not released.
- **Verify:** `make audit-verify` exits non-zero when the chain is broken. Recent events: `make audit-tail`.
- **Known limit:** the chain cannot detect truncation of the newest records. Copy the output of `python audit.py head` to an external store on a schedule to close that gap.
- **Review Citations**: Inspect expandable source citation drawers beneath responses.
- **Approve / Reject Outputs**: Handle compliance review checkpoints interactively when answers trigger review gates.

### Compliance CLI (`approve.py`)

Check and resume paused HITL checkpoint threads from the terminal. Every decision requires the reviewer's account and password, and the account must hold `review` on the thread's workspace:

```bash
# Inspect all threads currently waiting for human compliance
python approve.py --list

# Approve a paused thread (prompts for the reviewer's password)
python approve.py --thread ui-accounting-abc123 --action approved --approver bob --reason "Checked against policy"

# Reject a paused thread
python approve.py --thread ui-accounting-abc123 --action rejected --approver bob
```

---

## Repository Structure

```text
local-rag-lab/
├── app.py                   # LangGraph state machine, nodes, and Self-RAG conditional routes
├── workspace_manager.py     # Multi-tenant workspace lifecycle and storage isolation logic
├── ingest.py                # Multi-format document parser (.md, .txt, .pdf, .docx) & indexer
├── ui.py                    # Streamlit web application with citation drawers & HITL gates
├── approve.py               # CLI tool for inspecting and resolving HITL review threads
├── access.py                # Local accounts, password hashing, and per-workspace grants
├── audit.py                 # Hash-chained, append-only audit trail and verifier
├── eval.py                  # Offline evaluation gate over eval/corpus and eval/golden.json
├── export_graph.py          # Script generating Mermaid and PNG graph diagrams
├── requirements.txt         # Pinned top-level Python dependencies
├── Makefile                 # Unified setup, test, ingest, and execution workflows
├── Modelfile                # Local Ollama model parameters specification
├── docker-compose.yml       # Local PostgreSQL checkpoint database container config
├── pytest.ini               # Pytest configuration and test paths
├── .env.example             # Environment variable template
├── docs/                    # Architecture diagrams and benchmark documents
├── knowledge_base/          # Pre-packaged sample workspace document domains
└── tests/                   # Full test suite (engine, parsers, workspaces, integration)
```

---

## License

Apache-2.0
