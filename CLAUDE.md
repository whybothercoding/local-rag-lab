# Agent Guidelines & Architecture Reference

Standards for AI assistants working on **Local Enterprise RAG Lab**. Commands: `make help`. Measurements and decisions: `eval/README.md`, `docs/scaling.md`, `docs/lan-tls.md`.

---

## 1. Invariants

* **No external network at runtime** (`HF_HUB_OFFLINE=1`); the network is used only by `make models`.
* **Stack**: Python 3.11–3.14, LangGraph, Ollama (`qwen3:8b`, thinking off) or any OpenAI-compatible server, in-process ONNX embeddings + reranker + NLI chosen by `MODEL_PROFILE` (default `multilingual`; also `multilingual-large`, `en`), EN/DE/FR/EL (`rag_lab/lang/`), ChromaDB (default; Qdrant/pgvector opt-in), SQLite FTS5/BM25, SQLite or PostgreSQL, Streamlit, optional Rust accelerator (`crates/rag_accel`).
* **Fail closed**: Policy v3 RBAC, hierarchical document clearance, signed revocable session tokens, HMAC hash-chained audit log, optional TOTP for owners/admins, keys in the OS keychain (else owner-only files), AES-256-GCM sealing of state and cache.

---

## 2. Commands

Use `make` targets (`WS=`, never `WORKSPACE`). Once the first administrator exists, mutating user/workspace commands need `ADMIN=<admin>`.

```bash
make setup | models [PROFILE=...] | verify-models | status   # install, pinned models (network once), health
make lint                    # ruff
make typecheck               # mypy
make test                    # full suite; services optional, CI requires them
make test-offline            # same with sockets blocked
make eval                    # offline benchmark gates (eval/golden.json)
make graph                   # regenerate docs/architecture.mmd after editing graph edges
make add-user ACCOUNT=<u> [ADMIN=1] [ROLE=<owner|admin|reviewer|trusted|member|none>] [CLEARANCE=<level>] [AS=<admin>]
make set-role | grant | revoke | create-group | grant-group | set-clearance | team-mode | totp-enroll | recover ...   # see `make help`
make sync ADMIN=<admin> [WS=<ws>] [FORCE=1]   # index knowledge_base/ + uploads/
make ui [ADMIN=<admin>]
make up | down | db-password   # PostgreSQL on 127.0.0.1 (password in .env, created by `make up`)
make scale-up | scale-down SERVICE=<qdrant|pgvector|vllm>   # opt-in services; `make up` starts none
make clean | factory-reset [CONFIRM=yes]   # factory-reset also purges configured external stores
```

---

## 3. Architecture

Code is the `rag_lab/` package (`security/ retrieval/ ingestion/ storage/ pipeline/ ops/ ui/`); runtime data stays at the repo root (`rag_lab.ROOT`, override `RAG_LAB_HOME`). Configuration is one mutable `settings` object ([`config.py`](rag_lab/config.py)) read at call time: override fields, never copy them into module globals or default arguments. Run modules with `python -m rag_lab.<pkg>.<module>`.

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

`grade_hallucination` also exits to `human_approval` on `circuit_break`. Rewrites spend `MAX_RETRIEVAL_RETRIES`, regenerations `MAX_GENERATION_RETRIES` (with `grader_feedback` in the prompt). [`docs/architecture.mmd`](docs/architecture.mmd) is generated and authoritative; a test fails when it is stale.

* **State**: `asked_question` is the user's redacted literal question (reviewer, audit, UI); `original_question` the standalone form ([`context.py`](rag_lab/pipeline/context.py) rewrites follow-ups from *released* turns only, then re-scans and re-masks); cache key, retrieval, generation and relevance grading use it; `sanitized_question` is the expandable search query. One UI thread per conversation; input is disabled while a review is pending.
* **RBAC** ([`access.py`](rag_lab/security/access.py)): roles `owner/admin/reviewer/trusted/member/none` bundle `query/review/auto_approve` (`owner`/`admin` also administer). `effective_permissions` resolves: account's workspace grant, then its all-workspaces grant (`[]` denies), then groups' grants (any `none` wins), then own role, then the default role (`trusted` on a fresh install). Clearance is separate. Only authenticated administrators mutate accounts/roles/groups/grants/defaults/workspaces; only an owner assigns `owner`; the last owner stays. Older policy files migrate on load (old HMAC verified first, backup, `policy3:` MAC domain; `tests/test_roles.py` proves it lossless).
* **Keys and second factor** ([`keystore.py`](rag_lab/security/keystore.py)): a key file holds the key, an OS-keychain pointer, or an envelope-wrapped key (stub). `KEY_STORE=auto|file|keyring`; files move only via `make migrate-keys`; a pointer whose entry is gone is an error, never a new key. Tests pin `KEY_STORE=file` and an inert keyring: nothing may touch the real keychain. TOTP: sealed secret (AAD `uid:name`), code spent under the policy lock for a newer time step only (`_match_totp_step` is ours; pyotp has no `verify_and_get_timecode`). Lockouts: owners reset others; a sole owner is recovered only by `recover_account` (CLI, tty, audited, `LOCAL_RECOVERY=off` disables).
* **Sealed semantic cache** ([`semantic_cache.py`](rag_lab/retrieval/semantic_cache.py)): keyed `(workspace, clearance, index_fingerprint, original_question)`; rows are AES-GCM sealed; users without `auto_approve` read only reviewed entries. [`cache_index.py`](rag_lab/retrieval/cache_index.py) only proposes candidates: the winning row is re-read, opened, re-checked and its cosine recomputed. Never make the index the authority. `CACHE_BACKEND=postgres` shares the table (L1 off; same data key everywhere).
* **Document clearance**: unrecognized `clearance` tags are rejected at ingestion; dense and sparse retrieval filter by `allowed_clearances` and again with a fail-closed guard that drops hits without a valid clearance.
* **Vector stores**: all access goes through [`VectorStore`](rag_lab/storage/vector_store.py); `query()` runs `guard_hits()` after the adapter's server-side filter, so a new adapter cannot skip the guard. Similarity must stay cosine on Chroma's `1 - distance` scale (the gates are calibrated on it; compare values, not order). Qdrant/pgvector fail closed (no unfiltered fallback) and publish atomically. They store chunk text in the clear: RBAC is the application's job.
* **Ingest reuse**: a re-sync embeds only chunks whose exact text is new and reuses stored vectors of the same workspace, model fingerprint and `passage_prefix`; `FORCE=1` embeds everything. Plain-text files decode in-process; other formats stay in the sandboxed parser child.
* **Workspaces**: slugs `ws_<id>` merge `knowledge_base/<ws>` and `uploads/<ws>` (knowledge_base wins); `FORCE=1` swaps the Chroma collection atomically; deleting needs an admin and prunes grants.
* **Human review**: an answer releases automatically only if the user holds `auto_approve` and it is grounded, relevant, non-abstaining and free of blocking degradation; otherwise `interrupt()` waits for a reviewer with `review`, not the requester, whose clearance covers the thread's.
* **Graders** ([`grading.py`](rag_lab/pipeline/grading.py)): stated figures absent from the sources are rejected first, then every answer sentence must be entailed (`groundedness_min_entailment`) by the pinned NLI model in **full precision** — int8 weights scored one sentence 0.93 on arm64 and 0.12 on x86, so never pin quantized NLI weights. Without the model it falls back, labelled, to word overlap + cross-encoder.
* **Models**: every model is pinned by SHA-256 in [`model_integrity.py`](rag_lab/security/model_integrity.py). Thresholds are per profile and calibrated with `make calibrate-models`: changing a model means recalibrating, never reusing another model's numbers. Indexes record their `embedding_fingerprint`; a mismatch skips dense retrieval until `make sync FORCE=1`. Before pinning a quantized model, compare its scores across CPU architectures.
* **Degradation**: if dense search, reranking, the accelerator or the LLM is unavailable, fall back deterministically with labelled warnings; never fail silently. `RAG_ACCEL=preferred|required` (CI runs required); `make setup` installs a wheel from `wheelhouse/`, else builds with `cargo`, else uses the Python scanner. Injection patterns live in `rag_lab/security/injection_patterns.tsv`; rebuild with `make rust-build` after editing it.

---

## 4. Rules

* **Documentation sync**: `AGENTS.md` is a byte-identical copy of this file (`cp CLAUDE.md AGENTS.md`); a test fails otherwise. Keep both lean: rules and pointers here, measurements in `eval/README.md` and `docs/`.
* **Packaging**: metadata, dependencies, scripts and tool config live in [`pyproject.toml`](pyproject.toml); `requirements.txt` and `requirements.lock` (CI) must stay in step with it.
* **Access control**: every entry point validates identity through `principal_token` and workspace permissions via [`access.py`](rag_lab/security/access.py).
* **Sanitization and redaction**: scan queries and chunks for prompt injection ([`context_sanitizer.py`](rag_lab/security/context_sanitizer.py)), sanitize paths, enforce parser resource limits ([`parser_worker.py`](rag_lab/ingestion/parser_worker.py)), mask PII ([`redaction.py`](rag_lab/security/redaction.py)) before logs or prompts.
* **Testing**: every fix or feature needs tests in [`tests/`](tests/) and a green `make test`, `make test-offline`, `make lint`, `make typecheck`. Tests pin every scaling option off; a test that needs a service probes it and skips when it is unreachable (CI sets `REQUIRE_POSTGRES/QDRANT/PGVECTOR=1` and fails instead), because a URI derived from `.env` proves nothing. The UI is tested with Streamlit `AppTest`; assert `not app.exception`.
* **CI** (`.github/workflows`): `ci.yml` runs lint, mypy and the suite twice (online, offline) with PostgreSQL, Qdrant and pgvector services (about 20 minutes); `wheels.yml` builds the abi3 accelerator wheels. `ci.yml` runs on pushes and pull requests to `main`; `wheels.yml` when `crates/**` or the injection patterns change, and on `v*` tags.
