# Usage

Operational reference moved out of the README: accounts and roles, workspaces and ingestion, the web app, review, audit and governance, storage options and scaling.

## Setup

### 1. Prerequisites & Environment Setup

* **Python**: `>= 3.11` (tested up to 3.14)
* **Ollama** (optional, recommended for local generative LLM): `make ollama-pull` pulls the chat model (`qwen3:8b`, about 5 GB); embeddings, reranking and entailment run in-process, no Ollama needed. Any OpenAI-compatible server (vLLM, LM Studio, llama.cpp) works too: `LLM_PROVIDER=openai`, see [docs/scaling.md](scaling.md)
* **Rust Toolchain** (optional: only to build the native `rag_accel` extension when no prebuilt wheel fits your platform)
* **Docker** (optional: the PostgreSQL checkpointer, the Qdrant / pgvector / vLLM profiles, LAN TLS)

```bash
# 0. View all available make commands
make help

# 1. Create virtual environment and install dependencies
make setup

# 2. Download and verify the pinned embedding, reranker and NLI models (about 2 GB; requires network once)
make models

# 3. Verify cached model weights offline
make verify-models

# 4. (Optional) Build the native Rust accelerator from source. 'make setup' already installed a prebuilt wheel
#    from wheelhouse/ or built it when cargo exists; 'make status' says which, or that it is degraded.
make rust-build

# 5. Inspect full system status (models, accelerator, LLM, RBAC, workspaces, audit health)
make status
```

---

### 2. User Accounts & Role-Based Access Control (RBAC)

Nothing is reachable without a signed-in account. Passwords are hashed with **scrypt** with automatic hash upgrades.

**Roles.** Every account has a role, a named bundle of permissions: `owner` and `admin` (all permissions plus administration of accounts, groups, grants and workspaces), `reviewer` (query, review), `trusted` (query, auto_approve), `member` (query) and `none`. The first administrator is the `owner`; only an owner can assign or remove the owner role, and the last owner cannot be removed. An account without a role of its own follows the **default role**. Clearance is separate from roles.

**Groups** hold a role per workspace (or on `all` workspaces, including future ones), and accounts belong to groups. Permissions on a workspace resolve from the most specific source: the account's own grant on that workspace, then its own grant on all workspaces, then its groups' grants (combined, but any group's `none` denies), then its role, then the default role.

**Review modes.** A fresh install runs in **open** mode: the default role is `trusted` (`query` + `auto_approve`), so every account can chat on every workspace and answers that pass every automated check are released without a person reading them. Answers that fail a check still pause for review. For team use, appoint at least one reviewer and run `make team-mode ADMIN=<admin>`: it moves the default role from `trusted` to `member`, so answers wait for a reviewer unless an account holds `auto_approve` through its role, groups or grants. `make status`, `make users` and the admin panel show the current mode, and flag workspaces where a paused answer could not be decided (no reviewer, only the requester as reviewer, or no reviewer cleared as high as the requester), and warn when a shared install in open mode has confidential or secret documents indexed. Adding an account (from the second one on, you are asked for its role) prints the same warning when it leaves answers nobody can approve. As the team grows, `make status` suggests the next step (team mode, groups, a second owner); it never changes anything itself.

**Upgrading.** A policy file from an earlier version is migrated on first load, after its signature verifies: administrators become `owner` (the first alphabetically) or `admin`, the old defaults become the default role (or stay a custom permission set), and nobody's effective access changes. Where an administrator's new role would grant more than they held, their old access is pinned as an explicit grant on all workspaces. The old file is kept beside the new one as `access_policy.json.v2-<timestamp>.bak`, and the migration is written to the audit log. A file that fails its signature is refused and left untouched.

**Two-factor sign-in (optional, owners and administrators).** `make totp-enroll ADMIN=<you>` (or the sidebar panel) adds a TOTP secret to an authenticator app, confirms it with a code, and prints ten single-use recovery codes. From then on that account needs the code at every sign-in: the UI form, `--as` commands, `make sync`, and review sign-ins. The code is asked for only after the password verifies, a wrong code counts toward the lockout, and a code works once (replay-protected). The secret is stored sealed, bound to the account. Enforcement follows the enrolment, so a demoted administrator keeps it. Turn it off with `make totp-disable` (needs a code); an owner can reset another account's, which signs that account out. If no code is left to use, see *When someone is locked out* below: every case has an administrator-run fix, including a sole owner's.

**When someone is locked out.** Each case has a fix run by an administrator, or by whoever operates the machine:

| Situation | Who fixes it | How |
|---|---|---|
| Lost the authenticator, still has a recovery code | the user | sign in with the recovery code, then `make totp-recovery` or `make totp-disable` |
| Lost the authenticator and the codes: an administrator | an owner | `make totp-disable ADMIN=<owner> ACCOUNT=<them>` (or the owner's sidebar panel) |
| Same, for an owner, when another owner exists | the other owner | the same command |
| Same, for a sole owner | the machine operator | `make recover ACCOUNT=<them> RESET_TOTP=1` |
| Forgot the password: a member or an administrator | any administrator | `make set-password ACCOUNT=<them> ADMIN=<admin>` |
| Forgot the password: an owner | another owner | the same command (an administrator cannot reset an owner's password) |
| Forgot the password: a sole owner | the machine operator | `make recover ACCOUNT=<them> RESET_PASSWORD=1` |
| Locked out by five failed sign-ins | nobody needed | the lockout ends by itself within 15 minutes |
| Lost the access or data key | nobody | the policy and sealed data cannot be read without it: restore the key from custody (see *Where the keys live*) |

`make recover` is the break-glass: it needs no sign-in (the account cannot sign in), runs only from a terminal, asks you to type the account name back, resets the second factor and/or sets a new password, signs the account out everywhere, and writes `account_recovered_local` to the audit log. It exists on the command line only, never in the UI. It is not a new weakness: the key that signs the policy is readable only by the OS user running it, so anyone who can run it could already rewrite the policy by hand, without a trace. Restoring an old policy backup instead is a bad recovery plan, because it brings back deleted accounts and revoked sessions. On a shared host where the machine's operators are not all trusted, set `LOCAL_RECOVERY=off`; a sole owner's lockout then has no remedy.

**Where the keys live.** The access key (signs the policy and session tokens) and the data key (seals audit payloads, checkpoints and the cache, and keys the audit chain) go into the OS keychain when one is usable (macOS Keychain, Windows Credential Locker, Secret Service or KWallet); the key file then holds only a pointer. Where there is no keychain (headless Linux, containers, CI) they stay in owner-only files, and `make status` says so. Existing key files stay files until you run `make migrate-keys` (`REVERSE=1` moves them back). **`make backup` excludes keys, and a key held only in the keychain is lost with it (an OS reinstall, a keychain reset), taking the audit log, checkpoints and policy with it: keep a copy in custody** (`make migrate-keys REVERSE=1` writes the key files back). `KEY_STORE=auto|file|keyring` chooses the home of a new key; envelope encryption through Vault or a KMS has an interface and a stub (`rag_lab/security/keystore.py`) but no implementation. On macOS the keychain item belongs to the Python binary that created it: a rebuilt `venv`, or a different interpreter, can prompt or be refused, and a background process cannot answer a prompt (`KEY_STORE=file` avoids it).

**Audit anchors.** Set `AUDIT_ANCHOR_COPY_DIR` to a second location and every anchor is written to both; `make audit-verify` checks both and compares them (see the audit section).

All operations can be managed directly via `make` commands without activating the virtual environment:

```bash
# Create an administrator account (the owner; required for indexing and sync)
make add-user ACCOUNT=alice ADMIN=1

# Create accounts with a role and clearance level (public, internal, confidential, secret)
make add-user ACCOUNT=bob ROLE=reviewer CLEARANCE=internal ADMIN=alice
make add-user ACCOUNT=charlie ROLE=member CLEARANCE=public ADMIN=alice

# Change a role ('default' follows the default role) or a clearance level
make set-role ACCOUNT=charlie ROLE=trusted ADMIN=alice
make set-clearance ACCOUNT=bob CLEARANCE=confidential ADMIN=alice

# Groups: a role per workspace for everyone in the group
make create-group GROUP=finance-team ADMIN=alice
make add-to-group ACCOUNT=charlie GROUP=finance-team ADMIN=alice
make grant-group GROUP=finance-team WS=accounting ROLE=reviewer ADMIN=alice
make groups

# Per-account overrides on workspaces ('query', 'review', 'auto_approve', or a ROLE=)
# Accepts a single workspace, comma-separated list, or 'all'
make grant ACCOUNT=bob WS=accounting,strategy PERMS=query,review ADMIN=alice
make grant ACCOUNT=charlie WS=all ROLE=member ADMIN=alice

# Revoke access to workspaces (accepts a single workspace, comma-separated list, or 'all')
make revoke ACCOUNT=charlie WS=accounting ADMIN=alice
make revoke ACCOUNT=bob WS=all ADMIN=alice

# List accounts, roles, groups, clearance, and overrides (never displays hashes)
make users

# Require human review by default once a reviewer is appointed
make team-mode ADMIN=alice

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

# Sync a single workspace (auto-discovers knowledge_base/<ws>/ or accepts a custom DIR, which needs WS; optional FORCE=1)
make sync ADMIN=alice WS=accounting
make sync ADMIN=alice WS=accounting DIR=./knowledge_base/accounting FORCE=1
```

* Workspaces are automatically sanitized (e.g., `Accounting & Tax` becomes `accounting_tax`).
* Generates Chroma collections (`ws_<workspace_id>`) and SQLite FTS5 keyword indexes (`bm25_indices/<workspace_id>.db`).
* Idempotent: unchanged files are automatically skipped based on SHA-256 manifests (a parser or chunker change bumps `CHUNKER_VERSION`, so the next sync re-indexes everything once).
* Tables (DOCX, PPTX, XLSX, CSV) are written one row per line with the column names on every row (`Grade: G5 | Hotel cap EUR: 240`), so a chunk holding one row still says what each value is. DOCX keeps paragraphs and tables in document order and turns Heading styles into `#` lines, so chunks carry their section. PDFs are chunked page by page, so every chunk and citation carries its page (`annual.pdf (p. 12)`, the `page` metadata field); a sentence that crosses a page break is split there. A PDF with no text layer is refused with a note, and blank pages inside one are skipped with a warning.
* **Limits (by design, to stay simple and offline):** PDF extraction is plain text through `pypdf`: no OCR, no layout analysis, so tables inside a PDF flatten to running text and multi-column pages can read out of order. Headings are recognised in Markdown and DOCX only. Images, charts and legacy `.doc`/`.xls`/`.ppt` are not read. For those, run OCR first or export the table to XLSX/CSV.

---

### 4. Interactive Web Application

Launch the Streamlit web interface:

```bash
make ui ADMIN=alice
# Or launch without re-indexing (binds to 127.0.0.1 by default; LAN=1 to expose to network):
make ui
make ui LAN=1
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

# Export rejected answers as candidate eval cases for eval/golden.json (feedback loop)
make audit-harvest
make audit-harvest N=50 OUT=eval/feedback_candidates.json

# Anchor the audit head hash (run periodically to detect truncation; after a retention prune the new chain base carries a keyed integrity tag, so `make audit-verify` also flags a base edited to hide deleted records). Set AUDIT_ANCHOR_COPY_DIR to a second
# location (another disk, a mounted share): every anchor is written to both, `make audit-verify` checks both and
# compares them, so deleting or rewinding the anchor file on this host is detected. If the copy cannot be
# updated (an unmounted drive), the command exits non-zero; the primary anchor is still recorded.
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

# Zero-trace factory reset (wipes all users, audit logs, checkpoints, keys, vectors, and the source documents in knowledge_base/ and uploads/)
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

# Static gates
make lint
make typecheck

# Run offline RAG benchmark evaluation (eval/golden.json)
make eval

# Run evaluation with local Ollama LLM-as-a-judge support checks
make eval-judge

# Print retrieval similarity and keyword coverage signals on the calibration split
make eval-calibrate

# Export LangGraph architecture diagram to docs/architecture.mmd
make graph
```

About 1,000 tests. Services are optional locally: a test that needs PostgreSQL, Qdrant or pgvector probes it and skips when it is not running. GitHub Actions (`.github/workflows/ci.yml`) runs lint, mypy and the whole suite twice (online, then with sockets blocked) against real PostgreSQL, Qdrant and pgvector services and fails instead of skipping; a run takes about 20 minutes. The `wheels` workflow builds the Rust accelerator wheels.

---

### 8. Storage Architecture & Docker Evaluation

The engine supports two storage topologies for persisting LangGraph conversation state and checkpoints:

#### Default Mode: Embedded SQLite (Zero-Ops & Self-Contained)
* **Status:** Default out-of-the-box (`data/checkpoints.db`).
* **Operational Footprint:** **Zero Docker required.** Checkpoints, audit trails (`data/audit.db`), and FTS5 keyword indices run natively as embedded SQLite databases with Write-Ahead Logging (WAL) and AES-256-GCM encryption-at-rest.

#### Optional Mode: PostgreSQL via Docker Compose (`make up`)
The repository includes a `docker-compose.yml` to run PostgreSQL (`postgres:16-alpine`) as an external checkpointer (`DB_URI=postgresql://...`).

**Credentials.** There is no password in the repository. `make up` creates `.env` (mode 600, git-ignored) with a random `POSTGRES_PASSWORD` the first time; `docker-compose.yml` reads it through `${POSTGRES_PASSWORD}` and refuses to start without one. The server listens on `127.0.0.1` only. To use it as the checkpoint store, uncomment the `DB_URI` line in `.env` (it expands `${POSTGRES_USER}` and `${POSTGRES_PASSWORD}`). The PostgreSQL tests use `POSTGRES_TEST_URI`, or the credentials in `.env`.

**Scaling options (all off by default).** Qdrant or pgvector as the vector store, vLLM or any OpenAI-compatible server as the chat model, a PostgreSQL-backed semantic cache shared by several UI processes, each a compose profile started with `make scale-up SERVICE=<name>` that `make up` never starts: see [docs/scaling.md](scaling.md), including the measurements and what does **not** scale (the keyword index, the audit log and the policy stay local files).

**HTTPS on a LAN.** `make ui` serves plain HTTP and binds to `127.0.0.1` by default (opt-in for the network with `make ui LAN=1`). For colleagues on a network there is an optional Caddy proxy with a local CA (`make lan-tls`, a compose profile that `make up` never starts): see [docs/lan-tls.md](lan-tls.md).

**Upgrading from an earlier version.** The old compose file used a fixed password, and PostgreSQL applies `POSTGRES_PASSWORD` only when a volume is first created. After `make up` has created `.env`, run `make db-password` once: it sets the existing role's password to the one in `.env` over the container's local socket (the data is untouched). Starting fresh instead (`docker compose down -v`) deletes the volume and its checkpoints.

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

## 9. Scaling Beyond a Single Host: Cloud Architecture & Modern Stack

While **Local Enterprise RAG Lab** is engineered as a zero-cost, self-contained appliance for local workstations and air-gapped team servers, growing organizations often face requirements that justify migrating to a distributed cloud architecture.

**Already built, opt-in and off by default** ([docs/scaling.md](scaling.md)): Qdrant or pgvector as the vector store, vLLM or any OpenAI-compatible server for the chat model, and a PostgreSQL-backed semantic cache shared by several UI processes. The keyword index, the audit log, the policy and the keys remain local files. The rest of this section is a roadmap, not implemented.

### When & Why to Scale to the Cloud

1. **Document Volume & Index Memory Footprint (Beyond Single-Host RAM)**
   - *Local Constraint*: ChromaDB and SQLite FTS5 operate in-process. For corpora with tens of thousands of chunks (a few gigabytes), local memory is ideal. However, indexing millions of documents (tens of millions of chunks, terabytes of dense vector embeddings) exhausts single-host RAM and disk I/O.
   - *Cloud Driver*: Distributed vector databases partition and shard vector indexes (HNSW/IVF) across clustered nodes with distributed memory management and zero-downtime node rebalancing.

2. **High Concurrent Query Volume (Horizontal Autoscaling)**
   - *Local Constraint*: A single workstation or server comfortably handles 1 to 5 simultaneous queries. Under dozens or hundreds of concurrent users, embedding generation, cross-encoder reranking, and LLM token generation compete for CPU/GPU cores, causing latency spikes and queuing.
   - *Cloud Driver*: Containerized microservices running behind cloud load balancers auto-scale pod replicas horizontally (using KEDA based on request queue depth or CPU/GPU utilization), delivering steady sub-second p95 latencies under peak traffic.

3. **Frontier Models & Multi-GPU Infrastructure**
   - *Local Constraint*: Local inference is constrained to compact quantized models (e.g. `qwen3:8b`, `gemma4:12b`) running on CPU or consumer GPU memory.
   - *Cloud Driver*: Dedicated GPU node pools (e.g. NVIDIA H100, A100, L4) running high-throughput inference engines (**vLLM**, **TensorRT-LLM**, **Triton**) enable deploying 70B+ open models or privately proxying frontier LLMs (Claude 3.5 Sonnet, GPT-4o) with advanced multi-step reasoning.

4. **High Availability (HA), Multi-AZ Redundancy & 99.99% SLAs**
   - *Local Constraint*: A single local machine is a Single Point of Failure (SPOF). Power loss, hardware failure, or OS updates cause total downtime.
   - *Cloud Driver*: Multi-Availability Zone (Multi-AZ) clustering with managed database replication (e.g. AWS Aurora / GCP Cloud SQL), automated health checks, self-healing pod restarts, and Continuous Point-in-Time Recovery (PITR) to guarantee contractual enterprise uptime SLAs.

5. **Multi-Region Latency & Data Residency Compliance**
   - *Local Constraint*: All users connect to a single geographic location. Distributed remote teams experience high round-trip latency.
   - *Cloud Driver*: Deploying edge nodes behind Anycast routing (e.g. Cloudflare) serves queries close to users while enforcing strict geographical data boundaries (e.g., EU-only storage and inference for GDPR, US-only for HIPAA/ITAR).

6. **Continuous Event-Driven Ingestion from SaaS Ecosystems**
   - *Local Constraint*: Ingestion relies on batch triggers (`make sync`) or manual file uploads into workspace directories.
   - *Cloud Driver*: Real-time event streams (Apache Kafka, AWS SQS, Webhooks) connect directly to enterprise SaaS tools (Google Drive, SharePoint, Confluence, Jira, GitHub, Slack) to continuously parse, chunk, and index document edits as they happen via distributed worker workflows (Temporal, Celery).

7. **Enterprise SSO Federation & Centralized SIEM Auditing**
   - *Local Constraint*: Local scrypt passwords and HMAC-signed session tokens managed via `rag_lab/security/access.py`.
   - *Cloud Driver*: Direct federation with corporate Identity Providers (IdPs) via SAML 2.0 / OIDC (Okta, Microsoft Entra ID / Azure AD, Google Workspace), automated SCIM user provisioning, and real-time audit log streaming to enterprise SIEM platforms (Splunk, Datadog, AWS CloudWatch).

---

### Suggested Modern Cloud Stack Topology

To preserve this project's security invariants (Zero Trust clearance isolation, Self-RAG verification, four-eyes HITL review, and audit logging) while scaling horizontally, the following modern cloud architecture is recommended:

| Layer | Local Appliance (Current) | Cloud Enterprise Equivalent | Key Capabilities |
| :--- | :--- | :--- | :--- |
| **Orchestration & Compute** | Local Python runtime & Streamlit | **Kubernetes (EKS / GKE / AKS) + KEDA** | Horizontal Pod Autoscaling, stateless workers, rolling zero-downtime updates |
| **Vector Database** | Embedded ChromaDB (default); Qdrant or pgvector server, opt-in today | **Qdrant Cluster** or **pgvector on AWS Aurora** | Sharded HNSW indexing, billions of vectors, multi-tenant payload filtering |
| **Sparse Lexical Search** | SQLite FTS5 with BM25 ranking | **OpenSearch / Elasticsearch Cluster** | Distributed inverted index, cross-region search, tokenizers for all languages |
| **State & Checkpoints** | Embedded SQLite WAL (`checkpoints.db`) | **Managed PostgreSQL (Aurora Serverless / Cloud SQL)** | Multi-AZ failover, automated continuous backup (PITR), connection pooling |
| **Semantic & Response Cache** | SQLite float32 packed vector cache (shared PostgreSQL, opt-in today) | **Valkey / Redis Cluster with Vector Search** | Distributed sub-millisecond LRU cache with multi-node clearance isolation |
| **Model Inference Engine** | Local Ollama (`qwen3:8b`) or an OpenAI-compatible server such as vLLM (opt-in) / CPU ONNX | **vLLM / TensorRT-LLM on GPU Pools (A100/H100)** | Continuous batching, PagedAttention, 20x higher throughput, 70B+ models |
| **Ingestion Pipeline** | Batch CLI (`rag_lab/ingestion/ingest.py`) & child processes | **Temporal.io / Celery + Apache Kafka / SQS** | Distributed asynchronous parsing, rate-limited chunking, dead-letter queues |
| **Identity & Access** | Local `access_policy.json` (scrypt + HMAC) | **Okta / Microsoft Entra ID (OIDC / SAML 2.0)** | Enterprise SSO, SCIM automated provisioning, MFA enforcement |
| **Audit & Observability** | Tamper-evident HMAC SQLite log + anchors | **OpenTelemetry + Splunk / Datadog / CloudWatch** | Immutable cloud audit vaults (S3 Object Lock / WORM), distributed tracing |
| **User Interface & API** | Streamlit web UI & CLI | **FastAPI Async Gateway + React / Next.js SPA** | Server-Sent Events (SSE) streaming, WebSocket push, responsive web UX |

```mermaid
flowchart TD
    subgraph Clients["Clients & Ingress"]
        Web["Web Application (Next.js / React)"]
        CLI["Enterprise CLI / API Consumers"]
        Ingress["Cloud Load Balancer / API Gateway"]
        Web --> Ingress
        CLI --> Ingress
    end

    subgraph Auth["Identity & Access"]
        IdP["Enterprise IdP (Okta / Entra ID)"]
        Ingress -.->|OIDC / SAML| IdP
    end

    subgraph ServiceMesh["Container Mesh (Kubernetes / EKS)"]
        RAGService["RAG Orchestrator (LangGraph Agents)"]
        Ingress --> RAGService
        
        Cache["Valkey / Redis Cluster\n(Distributed Semantic Cache)"]
        RAGService <-->|Clearance-Filtered Cache| Cache
    end

    subgraph Retrieval["Distributed Storage & Search"]
        VectorDB["Qdrant / pgvector Cluster\n(Dense Embeddings)"]
        SearchDB["OpenSearch Cluster\n(Sparse BM25 Keyword)"]
        RAGService -->|Dense Vector Query| VectorDB
        RAGService -->|Sparse Lexical Query| SearchDB
    end

    subgraph Inference["Model Serving Infrastructure"]
        vLLM["vLLM / TensorRT-LLM GPU Cluster\n(LLM & Cross-Encoder Rerankers)"]
        RAGService -->|Inference & Scoring| vLLM
    end

    subgraph Storage["State & Compliance"]
        PG["Managed PostgreSQL (Aurora)\n(Checkpoints & Clearance Policy)"]
        AuditLog["Audit Vault (S3 Object Lock / WORM)\n+ Splunk / Datadog SIEM"]
        RAGService -->|State Persistence| PG
        RAGService -->|HMAC Hash-Chain Logs| AuditLog
    end

    subgraph AsyncIngest["Event-Driven Ingestion Pipeline"]
        SaaS["SaaS Connectors (Drive, Confluence, Git)"]
        Queue["Kafka / AWS SQS Event Queue"]
        Workers["Temporal / Celery Workers\n(Parser & Chunker Fleet)"]
        SaaS --> Queue
        Queue --> Workers
        Workers --> VectorDB
        Workers --> SearchDB
    end
```

