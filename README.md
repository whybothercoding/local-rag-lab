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

- **Python**: `>= 3.11` (Tested on 3.11–3.14)
- **Docker & Docker Compose**: For local PostgreSQL checkpoint storage
- **Ollama**: Installed and running locally (`http://localhost:11434`)

### 1. One-Command Setup

Run the automated setup target in the `Makefile` to create the virtual environment, install dependencies, spin up PostgreSQL, and pull the required model:

```bash
make setup
```

Alternatively, perform the steps manually:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch PostgreSQL checkpoint container
docker compose up -d

# 4. Pull LLM weights
ollama pull qwen2.5-coder:7b
```

### 2. Verify System Health

Run the full automated test suite (15 unit and integration tests covering engine nodes, workspace isolation, parser extractors, and database checkpoints):

```bash
make test
# Or: pytest -v
```

---

## Usage

### Ingesting Documents via CLI

Ingest folders of `.md`, `.txt`, `.pdf`, or `.docx` documents into separate isolated workspaces:

```bash
# Ingest pre-bundled sample domains
make ingest

# Or ingest custom directories into custom workspace bundles
python ingest.py --workspace accounting --dir ./knowledge_base/accounting
python ingest.py --workspace strategy --dir ./knowledge_base/strategy
```

Each workspace creates an isolated ChromaDB collection (`ws_<workspace_id>`) and a serialized BM25 index (`bm25_indices/<workspace_id>.pkl`).

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
- **Review Citations**: Inspect expandable source citation drawers beneath responses.
- **Approve / Reject Outputs**: Handle compliance review checkpoints interactively when answers trigger review gates.

### Compliance CLI (`approve.py`)

When running automated pipelines or backend workers, check and resume paused HITL checkpoint threads via terminal:

```bash
# Inspect all threads currently waiting for human compliance
python approve.py --list

# Review and approve a paused thread
python approve.py --thread ui-accounting-abc123 --action approve

# Reject a paused thread
python approve.py --thread ui-accounting-abc123 --action reject
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
