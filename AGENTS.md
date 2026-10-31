# AGENTS.md — Local Enterprise Knowledge Hub Specification

This document provides system instructions, architectural design details, and agent execution guidelines for the **Local Enterprise RAG Lab** Compound AI system.

---

## 1. System Overview

The engine is a fully local, multi-tenant Compound AI system implementing Self-RAG, Corrective RAG (CRAG), and Human-in-the-Loop (HITL) compliance gates. 

* **LLM Engine**: Ollama (`qwen2.5-coder:7b`) for generation and embeddings (`OLLAMA_BASE_URL=http://localhost:11434`).
* **Vector & Sparse Stores**: ChromaDB (dense vectors) + `rank-bm25` (sparse BM25Okapi).
* **State & Persistence**: LangGraph state graph with durable PostgreSQL checkpointer (`psycopg-pool`).
* **Multi-Tenancy**: Isolated workspace bundles for knowledge isolation and dynamic switching.

---

## 2. Multi-Tenant Workspace Architecture

Workspaces partition knowledge bases to prevent cross-tenant data leakage:

* **Sanitization**: Names are normalized via `sanitize_workspace_id()` to alphanumeric slugs with single underscores (e.g., `accounting_2026`).
* **ChromaDB**: Named vector collections prefixed with `ws_<workspace_id>`.
* **Sparse Index**: Pickled BM25 structures stored at `bm25_indices/<workspace_id>.pkl`.
* **Staging Area**: Uploaded raw documents stored under `uploads/<workspace_id>/`.
* **Semantic Cache**: State-keyed by workspace slug (`hashlib.sha256(f"{workspace_id}:{query}".encode())`).
* **Thread Tracking**: Persistent sessions isolated via `ui-<workspace_id>-<uuid>` thread IDs.

---

## 3. Supported Ingestion Pipeline (`ingest.py`)

The ingestion pipeline handles multi-format file parsing, character chunking, and synchronous index generation:

* **Supported Formats**:
  * `.md` & `.txt`: Raw UTF-8 extraction.
  * `.pdf`: Page-by-page extraction via `pypdf`.
  * `.docx`: Structural paragraph parsing via `python-docx`.
* **Chunking**: `RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)`.
* **Chunk Metadata**: Each chunk stores `{"source": filename, "workspace": ws_id, "id": f"{ws_id}_{doc_idx}_{chunk_idx}"}`.

---

## 4. LangGraph Agent Workflow

[START]
│
[guardrail] ──(fail)──> [END] (Prompt Injection / Adversarial)
│ (pass)
[cache_lookup] ──(hit)──> [END]
│ (miss)
[router] ──(chitchat)──> [chitchat] ──> [END]
│ (rag)
[retrieve] (Hybrid Dense + Sparse)
│
[grade_docs] ──(rewrite / max retries)──> [rewrite] ──> [retrieve]
│ (approved)
[generate]
│
[grade_hallucination] ──(retry)──> [generate]
│ (grounded)
[grade_relevance] ──(irrelevant)──> [rewrite]
│ (relevant)
[human_approval] (HITL Compliance Interruption)
│
[END]

### Graph State Schema (`GraphState`)
* `workspace_id` (`str`): Target workspace bundle name.
* `session_id` (`str`): Review-session identifier recorded with every audit event (`audit.py`).
* Audit payloads and checkpoint state are sealed at rest (`vault.py`). The audit chain is keyed, not a plain hash. Retention runs via `retention.py` (183 days by default).
* `degraded` (`List[str]`): Reduced-mode notes (unavailable indexes, model fallback) shown with the answer.
* `principal_token` (`str`): HMAC-signed session token from the UI or CLI. The guardrail derives `principal` from it, and a bare account name in the input is ignored.
* `principal` (`str`): Authenticated account, derived from `principal_token`. The guardrail denies it unless it holds `query` on `workspace_id`; the approval node requires `review` and rejects self-approval (`access.py`).
* `question` / `original_question` / `sanitized_question` (`str`): Query lifecycle states.
* `documents` (`List[str]`): Retrieved raw context chunks.
* `citations` (`List[str]`): Formatted provenance tags (`[chunk_id] source_file`).
* `generation` (`str`): Current response or compliance rejection text.
* `intent` (`str`): Routing classification (`rag` vs `chitchat`).
* `cache_hit` (`bool`): Direct semantic cache bypass flag.
* `is_safe` (`bool`): Input guardrail security gate.
* `grade_status` / `hallucination_status` / `relevance_status` (`str`): Self-RAG evaluation flags.
* `retry_count` (`int`): Retries counter with circuit breaker capped at 2.
* `history` (`List[str]`): Rolling conversation memory.
* `telemetry` (`Dict[str, Any]`): Latency, token metrics, and execution diagnostics.

---

## 5. UI & Traceability (`ui.py`)

* **Workspace Selector**: Sidebar selector dynamically scans directories and Chroma collections.
* **Bundle Creation & Deletion**: Staged file purge, Chroma collection deletion, and BM25 index cleanup.
* **In-App Ingestion**: Multi-file uploader supporting `.md`, `.txt`, `.pdf`, `.docx`.
* **Visual Citations**: Expandable citation cards (`render_citations()`) rendered below both streaming responses and past conversation turns.
* **HITL Action Drawer**: Halts on compliance approval gate with interactive review buttons.

---

## 6. Developer Workflows & Commands

```bash
# Create venv and install pure-Python dependencies (no Docker or model daemon required)
make setup

# Run the test suite (15 tests: engine, workspace manager, file parsers, integration)
make test

# Discover and index every workspace folder under knowledge_base/ (idempotent)
make sync

# Launch Streamlit interface
make ui
