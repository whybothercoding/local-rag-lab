# Design overview

The README is the short version. This file holds the design notes that used to sit in it: a self-assessment scorecard and the fourteen techniques the pipeline implements. Measurements and decisions are in [`eval/README.md`](../eval/README.md) and [`design-decisions.md`](design-decisions.md).

## Self-assessment scorecard

Scores are the author's self-assessment, not an external audit; each cites the mechanism and the measurement behind it.

| Area | Principle | Score | Architectural Implementation |
|:---|:---|:---:|:---|
| **Security** | **Explicit node verification** | **9 / 10** | Tokens, workspace permissions, and dual-control reviewer rules are enforced directly inside graph nodes, never delegated to client state. |
| **Security** | **Fail-Closed Retrieval** | **9 / 10** | Server-side clearance filtering backed by base-class `guard_hits()`; unknown clearances drop files; missing keys halt execution. |
| **Security** | **Tamper-Evident Audit** | **8.5 / 10** | Forward-secure HMAC-SHA256 hash chain with AES-256-GCM sealed payloads; signed head anchors replicated to off-host copies (`AUDIT_ANCHOR_COPY_DIR`). |
| **Security** | **Caller Isolation & DLP** | **8.5 / 10** | `prepare_pipeline_input()` sanitizes queries before state serialization; initial checkpoints and audit logs never persist unredacted PII. |
| **Security** | **Cryptographic Supply Chain** | **8.5 / 10** | Model weights pinned by hardcoded SHA-256 digests; all 164 package dependencies cryptographically hash-pinned in `requirements.lock`. |
| **Security** | **Network Boundary & Transport** | **8 / 10** | UI binds loopback `127.0.0.1` by default; network access requires explicit opt-in (`LAN=1`); local CA TLS via Caddy ([docs/lan-tls.md](lan-tls.md)). |
| **AI Systems** | **Compound Architecture** | **9 / 10** | Directed acyclic & cyclic LangGraph state machine with explicit retry budgets and decoupled retrieval/grading nodes. |
| **AI Systems** | **Reproducibility** | **9 / 10** | Pinned ONNX models; fp32 NLI yielding identical cross-entropy scores on arm64 and x86; deterministic extractive baseline. |
| **AI Systems** | **Doc & Code Fidelity** | **9 / 10** | Automated CI tests enforce that architecture diagrams, Makefile commands, and golden set metrics match codebase reality. |
| **AI Systems** | **Groundedness & Output Gates**| **8.5 / 10** | Quantitative figure extraction and cross-lingual NLI sentence entailment (`mDeBERTa-v3`) with self-corrective prompt feedback; claims that name what only a heading states are judged against the heading too. |
| **AI Systems** | **Tunability & Feedback Loop** | **9 / 10** | Parametric candidate retrieval widths (`RETRIEVAL_*_K`), measured on a 72-chunk near-duplicate stress suite (the defaults hold; a wider pool only doubles latency); automated harvesting of reviewer compliance rejections (`make audit-harvest`). |
| **AI Systems** | **Evaluation Benchmark** | **9 / 10** | 201 golden benchmark cases (143 held-out test split) across 4 languages, a messy-document suite (DOCX, XLSX, PDF) and a near-duplicate stress suite that measures the candidate pool with committed reference baselines, plus `make eval-llm` runs of the real model on every suite, which found a grader defect the deterministic run could not ([eval/README.md](../eval/README.md)). |

*Full security threat model and cryptographic boundaries: see [docs/threat-model.md](threat-model.md).*  
*Horizontal scaling and containerized cloud deployment options: see [docs/scaling.md](scaling.md).*

---

## Implemented techniques

These are the retrieval and grading patterns the lab implements and measures:

### 1. Compound AI Systems Architecture
Rather than treating LLM calls as a monolithic black box, the system orchestrates specialized, modular components: deterministic guardrails, semantic caching, intent routers, hybrid retrievers, document & hallucination graders, response generators, and compliance checkpoints.

### 2. Cyclic Agentic State Machines (LangGraph)
Uses **LangGraph** to model the RAG pipeline as a directed state graph (`GraphState`) featuring conditional branching, state cycles (retries and query rewriting loops), and durable execution interruptions.

### 3. Self-Reflective RAG (Self-RAG)
Implements dynamic self-assessment loops to verify output quality before answers are finalized:
* **Document Relevance Grading (`retrieve` + `grade_docs`)**: the evidence gate in `retrieve` (see Retrieval Abstention & Evidence Thresholds) keeps only chunks that clear the profile's relevance threshold; `grade_docs` then checks that any survived and otherwise routes to `rewrite`.
* **Hallucination Detection (`grade_hallucination`)**: Rejects figures the sources do not contain, then checks each answer sentence with a local, full-precision multilingual entailment (NLI) model (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, scores identical on arm64 and x86): a sentence passes only if a source sentence states it, in any of English, German, French, or Greek, including across languages. A rejected answer is regenerated with the unsupported sentence named.
* **Answer Relevance Grading (`grade_relevance`)**: Assesses whether the final answer directly addresses the original user prompt.

### 4. Corrective RAG (CRAG) & Adaptive Query Rewriting
When retrieved documents fail relevance grading or an answer is found irrelevant, the pipeline autonomously reformulates the user query (`rewrite`) and retries retrieval.

An answer that fails the groundedness check is regenerated from the same documents with the grader's reason added to the prompt (for example, the figures it stated that the context does not contain), because at temperature 0 an identical prompt would only repeat the same answer. The extractive fallback is never retried: it escalates straight to review. Two separate **circuit breakers** bound the loops: `MAX_RETRIEVAL_RETRIES` (default 2) for query rewrites and `MAX_GENERATION_RETRIES` (default 1) for regenerations, so one loop cannot spend the other's budget. Deployments running larger models can raise them in `.env`.

### 5. Hybrid Retrieval, GraphRAG & Cross-Encoder Reranking
Combines dense vector embeddings, sparse lexical search, in-process knowledge graph traversal, and neural cross-encoder reranking:
* **Dense Retrieval**: Cosine similarity via ChromaDB and in-process ONNX embeddings. `MODEL_PROFILE` picks the models: `multilingual` (default; `paraphrase-multilingual-MiniLM-L12-v2` + `bge-reranker-v2-m3-int8`, about 0.8 GB), `multilingual-large` (`multilingual-e5-large` + `bge-reranker-v2-m3`, about 4.5 GB, slower, marginally better), or `en` (`bge-small-en-v1.5` + `ms-marco-MiniLM-L-6-v2`, English only). Documents and questions may be in English, German, French, or Greek, in any combination.
  Models never switch automatically: each index records the model that built it, and retrieval skips dense search with a labelled warning until the workspace is re-indexed (`make sync FORCE=1`). `make status` marks such a workspace `[!!]`, and an answer built on it, including a "not found" abstention, waits for a reviewer instead of releasing on its own. Every profile also downloads the 1.1 GB NLI model (default profile: about 2 GB in all). There is no Ollama embedding path: it failed the multilingual gates, could not be hash-pinned and was slower (measurements in [eval/README.md](../eval/README.md)); a leftover `EMBEDDING_PROVIDER=ollama` fails with instructions.
* **Sparse Retrieval**: Lexical term matching via SQLite FTS5 / BM25 for exact matches, technical acronyms, and identifiers.
* **Graph-Augmented Retrieval (GraphRAG, experimental, off by default - `rag_lab/retrieval/graph_store.py`, `rag_lab/ingestion/graph_builder.py`)**: an in-process knowledge graph built during ingestion by regular expressions over English, German, French and Greek sentence patterns (*X is the Y*, *X requires approval from Y*, *X applies to Y*), traversed up to `GRAPH_MAX_HOPS=2` from entities named in the question, with an optional path viewer in the Streamlit UI. The intent is multi-hop bridging (Person to Role to System Authority to Access Ticket). **It has not been shown to work end to end:** on a 12-case multi-hop suite it raised the share of gold documents reaching the context from 0.361 to 0.556 but completed none of the 12 chains and changed no answer (5 of 12 contain the fact with graph retrieval off and on), because the extractor missed or mis-joined the later hops of ordinary prose. See the ablation in [eval/README.md](../eval/README.md). Enable with `ENABLE_GRAPH_RAG=1`.
* **Reciprocal Rank Fusion (RRF)**: Merges dense, sparse, and graph traversal candidate sets into an initial top candidate list.
* **Neural Cross-Encoder Reranker (`rag_lab/retrieval/reranker.py`)**: Shortlisted candidates undergo deep cross-attention re-scoring with a local, pinned ONNX model (per profile) to ensure that chunks directly answering the query are prioritized above partial keyword distractions while preserving graph-bridging evidence.

### 6. Retrieval Abstention & Evidence Thresholds ("Knowing When Not to Answer")
To prevent hallucinations on irrelevant context, the system enforces strict evidence thresholds:
* The gate is per model profile (calibrated with `make calibrate-models`). The default `multilingual` profile requires the cross-encoder reranker's best score to reach its threshold; the `en` profile requires cosine similarity $\ge 0.65$; either falls back to similarity or query keyword coverage when no reranker score exists.
* If retrieved chunks fail these criteria, the system cleanly **abstains** (explaining that documentation was not found) instead of fabricating answers from unrelated context. System abstention notices, clearance access denials, and compliance rejection banners are localized into Greek, German, French, and English, matching the question without language bleed.

### 7. State-Scoped & Clearance-Partitioned Semantic Caching
* Approved answers and their citations are stored sealed (AES-256-GCM) in a persistent SQLite cache with a bounded in-memory L1 in front (`CACHE_BACKEND=postgres` shares one table instead). Lookup is two-stage: an exact match on a composite key, `hashlib.sha256(f"{workspace_id}:{clearance}:{index_version}:{normalized_question}")`, then a cosine match inside the same partition, which also requires the same numbers, negations and language.
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
* Workflows can be resumed via either the web interface or terminal CLI (`rag_lab/ops/approve.py`).

### 11. Deterministic Source Provenance & Citations
Every answer sentence is tied back to traceable chunk keys (`[workspace_doc_chunk]`) and original source files, rendered in the UI with expandable citation cards for instant auditability.

### 12. Offline Benchmarking & LLM-as-a-Judge Evaluation Gate
An offline evaluation suite (`rag_lab/ops/eval.py`, `rag_lab/ops/eval_metrics.py`) benchmarks retrieval precision, keyword recall, groundedness, answer rate, and abstention accuracy against golden evaluation datasets (`eval/golden.json`), with optional LLM-as-a-judge support checks.

### 13. Multi-Tenant Knowledge Isolation & Document-Level RBAC
* **Workspace Isolation**: Workspaces (e.g., `accounting`, `legal`, `engineering`) are partitioned into isolated vector collections (`ws_<workspace_id>`) and independent SQLite FTS5 indices, preventing cross-tenant data leakage.
* **Document-Level Clearance Filtering**: Documents declare hierarchical clearance tags (`public`, `internal`, `confidential`, `secret`) in frontmatter or headers (unrecognized tags fail closed during ingestion). Vector search (`ChromaDB where` + post-query guard) and keyword search (`SQLite FTS5 WHERE clearance IN (...)`) strictly enforce the caller's authorized clearance, rendering classified documents completely invisible to unauthorized users.

### 14. Model Weight Pinning, Native Rust Acceleration & Graceful Degradation
* Embedding, cross-encoder reranker and NLI weights are verified against pinned SHA-256 hashes (`rag_lab/security/model_integrity.py`). The NLI model is pinned in full precision on purpose: its int8 sibling scored one true sentence 0.93 on Apple silicon and 0.12 on x86 Linux (details in [eval/README.md](../eval/README.md)).
* **Native Rust Accelerator (`crates/rag_accel`)**: the prompt-injection scan (`RegexSet` single-pass automaton, 24x faster than the Python regexes on 1,000 chunks: 0.87 ms against 21.2 ms; a later best-of-three re-run measured 0.37 ms against 14.4 ms, 39x, so the factor varies with the machine and the run) with verdict and reason parity, from the same pattern file as the Python scanner. It is the default, with a fallback that is never quiet: `RAG_ACCEL=preferred` (default) uses it when installed and otherwise falls back to the pure-Python scanner with one warning on stderr, and `make status` shows `[!!] DEGRADED`; `RAG_ACCEL=required` refuses to start without it. `make setup` installs it for you, in order: a prebuilt wheel from `wheelhouse/`, a build from source when `cargo` exists, else the fallback. The `wheels` GitHub workflow builds one abi3 wheel per platform (macOS arm64 and x86_64, Linux x86_64 and aarch64) that serves Python 3.11 and later; download the artifact for your platform into `wheelhouse/` and run `make setup`. There is no native chunker: chunking was 0.05 s of a 91 s ingest.
* If the primary LLM, cross-encoder, or vector store is unreachable, the system enters a degraded mode with transparent extractive fallbacks and visible warnings rather than failing silently.
* **Measured performance** (Apple M-series, 150 documents / 4,055 chunks, idle machine, before to after). Ingest: a full sync 79.6 s to 69.9 s; a re-sync with nothing changed 7.97 s to 0.02 s; a re-sync after editing one file 78.8 s to 3.7 s. Re-syncs embed only chunks whose text is new and reuse the stored vector of every other chunk (same workspace, same model, same prefix; `FORCE=1` recomputes everything), and plain-text files are decoded in-process instead of in a sandboxed child (the sandbox stays for every other format). Embedding is over 80% of a full sync and runs at about 55 to 60 chunks/s however it is batched: one batch size, thread count or worker-process setting after another was no faster. Query: a question that misses a cache of 2,000 answers spent 217 ms in the cache before and 5 ms now (57.5 ms to 4.1 ms at 500), so such a query takes 243 ms instead of 476 ms; with an empty cache nothing changes. The semantic cache searches one matrix per (workspace, clearance, index version) partition, only the clearances the caller may read, and re-reads and re-verifies the winning row before it answers.

---

## Repository structure

```text
local-rag-lab/
├── pyproject.toml           # Declarative PEP 517/518/621 package & tool configurations
├── requirements.txt / .lock # Runtime dependencies (the lock file is what CI installs)
├── ui.py                    # Streamlit launcher (runs rag_lab/ui/streamlit_app.py)
├── rag_lab/                 # The package; data stays at the repo root (rag_lab.ROOT / RAG_LAB_HOME)
│   ├── config.py            # One mutable `settings` object built from the environment, read at call time
│   ├── pipeline/            # LangGraph Self-RAG: state, entry, retrieval, generation, grading, review, graph
│   │                        #   + llm_provider (Ollama / OpenAI-compatible / extractive fallback)
│   ├── retrieval/           # Embeddings, FTS5 keyword index, reranker, NLI, semantic cache + its vector index
│   ├── ingestion/           # Indexer, format parsers, sandboxed parser worker
│   ├── security/            # Accounts & RBAC, keystore, AES-GCM vault, hash-chained audit, redaction, injection scan, model pins
│   ├── storage/             # VectorStore (Chroma, Qdrant, pgvector), checkpointer, workspaces, backups, retention
│   ├── lang/                # EN/DE/FR/EL text layer and language data
│   ├── ops/                 # approve (HITL CLI), eval + metrics, calibration, model download, accelerator install, purge
│   └── ui/                  # Streamlit page
├── crates/rag_accel/        # Native Rust PyO3 accelerator (prompt-injection scan), built as one abi3 wheel per platform
├── Makefile                 # Unified setup, test, sync, UI, review, and audit workflows (`make help`)
├── docker-compose.yml       # PostgreSQL for durable checkpoints; opt-in profiles: qdrant, pgvector, vllm, caddy
├── deploy/                  # Caddyfile for the optional LAN TLS proxy
├── docs/                    # design-overview.md, design-decisions.md, usage.md, scaling.md, lan-tls.md, threat-model.md, architecture.mmd
├── .github/workflows/       # ci.yml (lint, mypy, tests online + offline), eval-llm.yml (weekly, non-blocking), wheels.yml
├── eval/                    # Golden test dataset, benchmark corpus, measurements and decisions (README.md)
├── knowledge_base/          # Multi-tenant knowledge base folders
└── tests/                   # ~1,000 unit, integration and security tests
```
