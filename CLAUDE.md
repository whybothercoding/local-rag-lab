# Agent Guidelines & Architecture Reference

Instructions and engineering standards for AI assistants and coding agents working on the **Local Enterprise RAG Lab** codebase.

---

## 1. Project Overview & Invariants

Local Enterprise RAG Lab is a fully local, air-gapped Compound AI retrieval and compliance engine.

* **No External Network**: Inference and pipelines run 100% offline (`HF_HUB_OFFLINE=1`). Network access is permitted only during initial model download (`make models`).
* **Core Stack**: Python 3.11–3.14, LangGraph, Ollama (`qwen3:8b`, thinking off), ONNX FastEmbed + cross-encoder reranker chosen by `MODEL_PROFILE` (default `multilingual`: `paraphrase-multilingual-MiniLM-L12-v2` + `bge-reranker-v2-m3-int8`; also `multilingual-large`, `en`), languages EN/DE/FR/EL (`rag_lab/lang/`), ChromaDB, SQLite FTS5 / BM25, SQLite / PostgreSQL (`psycopg-pool`), Streamlit, and an optional Rust PyO3 accelerator (`crates/rag_accel`).
* **Fail-Closed Security**: Policy v3 RBAC (roles `owner`/`admin`/`reviewer`/`trusted`/`member`/`none`, groups) with out-of-the-box default role `trusted` (`query` + `auto_approve`), admin-only user/group/workspace management (`--as <admin>`), hierarchical document clearance (`public`, `internal`, `confidential`, `secret`), revocable signed session tokens (`principal_token` bound to `uid` + `token_version`), tamper-evident HMAC hash-chained audit logging (anchors optionally copied to a second location), optional TOTP second factor for owners/admins, keys in the OS keychain (else owner-only files), and thread-bound AES-256-GCM state and cache encryption at rest (`ENC2` / `SEAL1`).

---

## 2. Essential Commands

Always use `make` targets (and `WS` for workspace arguments, never `WORKSPACE`). Mutating user and workspace commands require `ADMIN=<admin>` once the first administrator is created:

```bash
# Setup, Models & Health
make help                   # Show categorized command reference (default target)
make status                 # End-to-end health check (models, accelerator, LLM, RBAC, workspaces, audit)
make setup                  # Create venv and install Python dependencies
make models [PROFILE=...]   # Download & verify the profile's pinned ONNX models (requires network once)
make verify-models          # Verify model SHA-256 integrity offline
make calibrate-models       # Suggest a profile's thresholds from the eval calibration split
make rust-build             # Build the native Rust PyO3 accelerator (rag_accel) from source; `make setup` installs a wheel from wheelhouse/ first

# RBAC & User Management (Admin-Gated)
make add-user ACCOUNT=<u_name> [ADMIN=1] [ROLE=<role>] [CLEARANCE=public|internal|confidential|secret] [AS=<admin>]
make set-role ACCOUNT=<u_name> ROLE=<owner|admin|reviewer|trusted|member|none|default> ADMIN=<admin>
make grant ACCOUNT=<u_name> WS=<ws|ws1,ws2|all> [PERMS=query,review,auto_approve | ROLE=<role>] ADMIN=<admin>
make revoke ACCOUNT=<u_name> WS=<ws|ws1,ws2|all> ADMIN=<admin>
make set-default ROLE=<member|trusted|reviewer|none> ADMIN=<admin>  # or PERMS=<list> (custom set)
make team-mode ADMIN=<admin>  # Default role trusted -> member (open -> team review mode); needs a reviewer
make set-admin ACCOUNT=<u_name> ADMIN=<admin> [REVOKE=1]  # shortcut for ROLE=admin / ROLE=default
make groups | create-group GROUP=<g> | delete-group GROUP=<g> ADMIN=<admin>
make add-to-group | remove-from-group ACCOUNT=<u_name> GROUP=<g> ADMIN=<admin>
make grant-group GROUP=<g> WS=<ws|ws1,ws2|all> ROLE=<reviewer|trusted|member|none|default> ADMIN=<admin>
make set-clearance ACCOUNT=<u_name> CLEARANCE=<level> ADMIN=<admin>
make totp-enroll | totp-recovery ADMIN=<me>; make totp-disable ADMIN=<me> [ACCOUNT=<u>]  # TOTP for owners/admins; reset of others: owner only
make recover ACCOUNT=<u> [RESET_TOTP=1] [RESET_PASSWORD=1]  # operator break-glass, no sign-in, tty + typed confirmation, audited; LOCAL_RECOVERY=off disables
make migrate-keys [REVERSE=1]  # access/data keys: files <-> OS keychain (backup excludes keys)
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
make typecheck              # mypy gate on the rag_lab package
make coverage               # Test suite with per-module coverage report
make eval                   # Run offline RAG benchmark evaluation (eval/golden.json)
make eval-judge             # Evaluation with local Ollama LLM-as-a-judge support checks
make eval-llm [SUITE=en]    # Answers from the local Ollama model; judge support gated
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
make up | down | db-password   # PostgreSQL (127.0.0.1, password in .env, created by `make up`); db-password applies it to an old volume
make lan-tls | lan-tls-down | lan-tls-ca   # optional Caddy HTTPS for the LAN (docs/lan-tls.md)
make scale-up | scale-down SERVICE=<qdrant|pgvector|vllm>   # optional scaling services (docs/scaling.md); `make up` starts none
```

---

## 3. Architecture & LangGraph Pipeline

Code lives in the `rag_lab/` package (`security/`, `retrieval/`, `ingestion/`, `storage/`, `pipeline/`, `ops/`, `ui/`); runtime data stays at the repo root (`rag_lab.ROOT`, override with `RAG_LAB_HOME`). Configuration is one mutable `settings` object ([`rag_lab/config.py`](rag_lab/config.py)) read at call time: override fields, never copy them into module globals or default arguments. Run modules with `python -m rag_lab.<pkg>.<module>`.

Graph wiring is in [`rag_lab/pipeline/graph.py`](rag_lab/pipeline/graph.py); stages live in `entry`, `retrieval`, `generation`, `grading`, and `review`. It follows a cyclic Self-RAG state machine:

```text
[START] ──> [guardrail] ──(fail)──> [END]
                 │ (pass)
          [contextualize]   (follow-up -> standalone question)
                 │
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
* **State Schema**: `GraphState` ([`state.py`](rag_lab/pipeline/state.py)) keeps the user's literal redacted question in `asked_question` (reviewer, audit, UI) and the standalone form in `original_question`: [`contextualize_node`](rag_lab/pipeline/context.py) rewrites follow-ups using only *released* turns from `history` (`{question, asked, answer, released}`), then re-scans and re-masks the rewrite. Cache key, retrieval, generation, and relevance grading use `original_question`; `sanitized_question` is the search query that [`query_rewrite_node`](rag_lab/pipeline/retrieval.py) may expand. The UI keeps one thread per conversation and disables input while it awaits review (new input on a paused thread would abandon the pending review).
* **Policy v3 RBAC Resolution**: roles are named bundles of the workspace permissions (`ROLE_PERMISSIONS`); `owner`/`admin` also carry administration (`is_admin` reads the role only). [`effective_permissions`](rag_lab/security/access.py) resolves in priority order: (1) the account's explicit workspace grant (`user["grants"][ws]`), then its all-workspaces grant (`user["default_permissions"]`, set via `grant <user> all`); `[]` is an explicit denial, (2) its groups' grants (`policy["groups"][g]["grants"][ws or "*"]` = role; union, any `none` wins), (3) its own `role`, (4) the default role (`policy["defaults"]`: `{"role"}` or a custom `{"permissions"}`; `trusted` on a fresh install). Clearance is separate. Only authenticated administrators (`actor` / `--as <admin>`) can mutate users, roles, groups, grants, clearances, defaults, or workspaces; only an owner assigns/removes `owner`, and the last owner stays. v1/v2 files migrate on load (old HMAC verified first, backup written, `policy3:` MAC domain, access pinned where a role would widen it; `tests/test_roles.py` proves it lossless).
* **Keys & Second Factor**: [`keystore.py`](rag_lab/security/keystore.py) owns where the access and data keys live: a key file holds the raw key, a pointer to an OS-keychain entry (random account per key), or an envelope-wrapped key (interface + stub only). `KEY_STORE=auto|file|keyring`; existing files move only via `make migrate-keys`; a pointer whose entry is gone is an error, never a new key; keychain keys are cached per process. Tests pin `KEY_STORE=file` and an inert keyring backend: nothing may reach the real keychain. TOTP (`access.py`): self-enrolment in two steps, secret sealed with AAD `uid:name`, code spent under the policy lock for a newer time step only, recovery codes stored as keyed tags, enforced for as long as the enrolment exists. Lockouts: an owner's password and second factor are reset by an owner (`set_password`, `disable_totp`), a sole owner's by `recover_account` (CLI only, README table "When someone is locked out"). pyotp 2.10 has no `verify_and_get_timecode`: `_match_totp_step` is ours.
* **Clearance-Partitioned Sealed Cache**: `SEMANTIC_CACHE` and `SEMANTIC_CACHE_CITATIONS` ([`semantic_cache.py`](rag_lab/retrieval/semantic_cache.py)) provide an LRU in-memory + AES-256-GCM sealed SQLite-persistent cache keyed on `(workspace_id, clearance, index_fingerprint, original_question)` with cosine-similarity semantic retrieval, numeric/negation signature verification, and `reviewed` status tracking so users without `auto_approve` only consume human-reviewed cache entries. Semantic search is [`cache_index.py`](rag_lab/retrieval/cache_index.py): one float32 matrix per (db, workspace, clearance, index version) partition; a lookup searches only the caller's authorized clearances, re-syncs each partition with the table by `(key, rowid, created_at, reviewed)` first, and the winning row is re-read, opened (seal + AAD), re-checked and its cosine recomputed before a hit. The index only proposes candidates; never make it the authority.
* **Fail-Closed Document Clearance**: Unrecognized explicit `clearance` tags in YAML frontmatter (case-insensitive) or `.clearance` folder files are rejected during ingestion ([`ingest.py`](rag_lab/ingestion/ingest.py)). Both dense ([`retrieval.py`](rag_lab/pipeline/retrieval.py)) and sparse ([`sparse_search.py`](rag_lab/retrieval/sparse_search.py)) retrieval enforce `allowed_clearances` at query time and via fail-closed post-query guards that reject chunks missing a valid clearance attribute.
* **Scaling options (off by default)**: all vector access goes through [`VectorStore`](rag_lab/storage/vector_store.py) (`VECTOR_STORE=chroma|qdrant|pgvector`); `query()` applies `guard_hits()` after the adapter's server-side filter, so the clearance guard lives in the base class and a new adapter cannot skip it. Similarity must stay cosine on Chroma's `1 - distance` scale (the gates are calibrated on it: tests compare values, not just order). Qdrant/pgvector fail closed (no unfiltered fallback) and publish atomically. `CACHE_BACKEND=postgres` puts the sealed semantic cache in a shared database (L1 off; same data key everywhere). `LLM_PROVIDER=openai` (`LOCAL_LLM_*`) serves vLLM/LM Studio/llama.cpp: thinking off via `chat_template_kwargs`, `<think>` stripped. Tests pin all of it off (conftest, eval redirect) so a `.env` can never make a test reach a real service; adapters are tested in their own namespace (`QDRANT_PREFIX`, `PGVECTOR_SCHEMA`).
* **Ingest reuse**: a re-sync embeds only chunks whose exact text is new and takes the stored vector of the rest, but only from the same workspace's live collection with the same model fingerprint (collection and manifest) and the same `passage_prefix` recorded in the manifest; `FORCE=1`, an unreadable index or an older manifest embeds everything. Plain-text files (`PLAIN_SUFFIXES`) are decoded in-process; every other format stays in the sandboxed parser child.
* **Workspaces**: Sanitized slugs (`ws_<workspace_id>`). Each workspace deterministically merges `knowledge_base/<ws>` and `uploads/<ws>` (with `knowledge_base` taking precedence on filename collisions) and performs an atomic Chroma collection swap (`__build` -> live) during `FORCE=1` re-syncs. Deleting a workspace requires admin privileges, escapes SQL `LIKE` wildcards, and prunes its grants from `access_policy.json`.
* **Human-in-the-Loop & Safety Escalation**: Answers release automatically when the user holds `auto_approve` **and** the generation is grounded, relevant, non-abstaining, and free of blocking degradation (`security_warning`, `dense_retrieval_unavailable`, `sparse_retrieval_unavailable`). Otherwise, execution pauses at `interrupt()` for a reviewer who holds `review` permission, is not the requester (`requester != approver`), and whose clearance dominates (`clearance_covers`) the thread's `principal_clearance`.
* **Self-RAG Graders**: [`grade_hallucination_node`](rag_lab/pipeline/grading.py) first rejects stated figures absent from the sources (`_unsupported_numbers`), then requires every answer sentence to be entailed (>= `groundedness_min_entailment`) by a pinned multilingual NLI model ([`nli.py`](rag_lab/retrieval/nli.py), `mDeBERTa-v3-base-mnli-xnli`), premise = single source sentences plus neighbour pairs that share the claim's vocabulary; "the documents don't mention X" sentences are skipped, and the rejected sentence goes into the retry feedback. Without the NLI model it falls back, labelled, to word overlap + cross-encoder. [`grade_answer_relevance_node`](rag_lab/pipeline/grading.py) uses the reranker's cross-encoder.
* **Rust Accelerator Parity & Graceful Degradation**: `crates/rag_accel` (PyO3 0.30, abi3: one wheel per platform for Python 3.11+, built by `.github/workflows/wheels.yml`) strips zero-width Unicode obfuscation and runs a single-pass `RegexSet` automaton with 100% verdict and reason parity with [`context_sanitizer.py`](rag_lab/security/context_sanitizer.py); there is no native chunker (0.05% of ingest). [`rag_lab/accel.py`](rag_lab/accel.py) owns `RAG_ACCEL=preferred|required`: a missing or failing accelerator is a one-time stderr warning and `make status` `[!!] DEGRADED` (preferred) or `AcceleratorUnavailableError` at import of the sanitizer (required; CI runs required). `make setup` runs `rag_lab/ops/install_accel.py`: installed, else a wheel in `wheelhouse/`, else `cargo`, else the Python fallback. If dense search, cross-encoder reranking, or the local LLM is unavailable, fall back deterministically with labeled warnings; never fail silently. Never switch embedding models automatically: indexes store their `embedding_fingerprint`, and a mismatch skips dense retrieval until re-index. Thresholds are per model profile (`rag_lab/config.py`), calibrated with `make calibrate-models`; changing a model means recalibrating, never reusing another model's numbers. Injection patterns live in one file (`rag_lab/security/injection_patterns.tsv`) that Rust compiles in: rebuild `make rust-build` after editing it.

---

## 4. Coding & Security Rules

* **Documentation Sync**: Keep `AGENTS.md` and `CLAUDE.md` identical at all times.
* **Packaging & Configuration Standards**: Project builds, metadata, dependencies, console scripts, and tool configs (pytest, ruff, mypy) are centrally declared in [`pyproject.toml`](pyproject.toml) following PEP 517/518/621. Editable installs (`pip install -e . --no-build-isolation`) register standard console scripts (`local-rag`, `local-rag-sync`, `local-rag-access`, etc.).
* **Access Control**: All entry points must validate caller identity through `principal_token` and workspace permissions (`query`, `review`, `auto_approve`) and roles (`owner`/`admin` administer) via [`access.py`](rag_lab/security/access.py).
* **Sanitization & Redaction**: Normalize and scan both user queries and document chunks for prompt injection ([`context_sanitizer.py`](rag_lab/security/context_sanitizer.py)), sanitize file paths, enforce zip-bomb/PDF resource limits in [`parser_worker.py`](rag_lab/ingestion/parser_worker.py), and mask PII (emails, phones, SSNs, Luhn payment cards, mod-97 IBANs in [`redaction.py`](rag_lab/security/redaction.py)) before persisting logs or prompt contexts.
* **Testing**: Any bug fix or feature must include unit/integration coverage in [`tests/`](tests/) and pass `make test`. PostgreSQL tests use the `postgres_uri` fixture: they run against `POSTGRES_TEST_URI`, or the docker-compose server when `make up` is running (a recreated `<db>_pytest` database, also in the offline run when local), and CI fails them instead of skipping (`REQUIRE_POSTGRES=1`). The UI is tested with Streamlit `AppTest`; assert `not app.exception`.
