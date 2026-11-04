# Scaling options (all off by default)

Nothing here is needed on one laptop: `make setup && make models && make ui` uses embedded Chroma, a SQLite semantic
cache, and Ollama or the in-process synthesizer. Each option below is opt-in, starts nothing on its own, and can be
turned off again by deleting one line of `.env`. Read *What does not scale* before you plan a multi-host deployment.

| Option | Turn it on | What it gives you | Cost |
|---|---|---|---|
| Vector store: **Qdrant** | `make scale-up SERVICE=qdrant`, `VECTOR_STORE=qdrant` | One vector index shared by several processes or hosts; 3x faster queries and 4x faster publishes than embedded Chroma at 20,000 chunks | A server (292 MB image), `pip install qdrant-client`, a re-sync |
| Vector store: **pgvector** | `make scale-up SERVICE=pgvector`, `VECTOR_STORE=pgvector` | The same, inside PostgreSQL you may already run and back up | A server (640 MB image), a re-sync; exact scan, no ANN index |
| Chat model: **vLLM** / any OpenAI-compatible server | `LLM_PROVIDER=openai`, `LOCAL_LLM_URL=...` (`make scale-up SERVICE=vllm` on a GPU host) | Throughput for many concurrent users; a bigger model than one laptop runs | An NVIDIA GPU for vLLM; the model is not hash-pinned |
| **Shared semantic cache** | `CACHE_BACKEND=postgres`, `CACHE_DB_URI=...` | Several UI processes see one cache, sealed as before | A PostgreSQL database; a few conditions on keys and indexes (below) |
| **LAN TLS** | `make lan-tls` | HTTPS in front of the UI | See [lan-tls.md](lan-tls.md) |

`make scale-up SERVICE=<qdrant|pgvector|vllm>` starts one compose profile; `make scale-down SERVICE=...` stops it and keeps
its data. `make up` never starts any of them. Every service binds `127.0.0.1`.

## Vector store

Chroma stays the default. `VECTOR_STORE=qdrant` or `pgvector` swaps where chunk vectors live, behind one interface
(`rag_lab/storage/vector_store.py`), and **nothing else in the pipeline changes**. Switching does not migrate: the old
index is left where it was, and `make sync ADMIN=<admin> FORCE=1` builds the new one.

```bash
pip install 'qdrant-client>=1.9'        # Qdrant only; pgvector needs no extra package
make scale-up SERVICE=qdrant            # or pgvector (reuses POSTGRES_USER / POSTGRES_PASSWORD from .env, port 5433)
# .env:  VECTOR_STORE=qdrant   QDRANT_URL=http://localhost:6333     (pgvector: VECTOR_STORE=pgvector)
make sync ADMIN=<admin> FORCE=1
make status
```

Settings: `QDRANT_URL`, `QDRANT_API_KEY` (empty means no authentication: fine on loopback, set one before the port is
reachable from anywhere else), `QDRANT_PREFIX` (prepended to collection names so installs can share a server),
`PGVECTOR_URI` (default: derived from `POSTGRES_*` for the compose profile), `PGVECTOR_SCHEMA` (default `public`).

What every store guarantees, and the tests that pin it (`tests/test_vector_stores.py`, run against Chroma, embedded
Qdrant, a real Qdrant server and a real pgvector server):

* **The clearance guard is in the base class.** Each store filters by clearance on the server, and then
  `VectorStore.query()` drops any hit with a missing or invalid clearance, one the caller may not read, or another
  workspace's. A broken filter in an adapter cannot leak a chunk. A test with a deliberately leaky adapter proves it.
* **Similarity is cosine on Chroma's scale.** The relevance gates are calibrated on it, so the values are compared, not
  just the order (within 1e-4 of Chroma on the same corpus). `make eval` passes through each real adapter with the same
  numbers as Chroma (answer rate and retrieval recall 0.9773, abstention accuracy 1.0).
* **A publish is atomic.** Qdrant builds a new collection and moves an alias in one request; pgvector replaces the
  workspace's rows in one transaction. A publish that fails part-way leaves the old index serving.
* **No unfiltered fallback.** If the filter fails, dense retrieval becomes unavailable (labelled) instead of asking for
  everything and trusting the guard. Chroma keeps its old fallback for old collections, behind the same guard.

Measured on this machine, 384-dimensional random vectors, top 5 with a clearance filter, servers on loopback (one run;
random vectors are the worst case for an HNSW build):

| Chunks | Store | Publish | Query median (p95) |
|---:|---|---:|---:|
| 4,000 | Chroma (embedded) | 1.64 s | 8.5 ms (11.5) |
| 4,000 | Qdrant | 0.76 s | 6.9 ms (11.9) |
| 4,000 | pgvector (exact) | 0.88 s | 3.5 ms (6.7) |
| 20,000 | Chroma (embedded) | 11.9 s | 31.9 ms (37.0) |
| 20,000 | Qdrant | 3.0 s | 9.6 ms (13.0) |
| 20,000 | pgvector (exact) | 4.3 s | 12.0 ms (15.4) |

A whole query takes about 240 ms, and embedding the question is 4.5 ms of it: at the sizes this project is built for the
stores differ by single milliseconds. Choose one to **share an index**, not for speed.

pgvector creates no ANN index: the column has no fixed dimension (the model can change) and HNSW needs one. The exact scan
grows linearly; beyond a few hundred thousand chunks per workspace, add an HNSW index on the cast column yourself.

## Chat model: vLLM or any OpenAI-compatible server

`LLM_PROVIDER=openai` talks to any server that implements `POST /v1/chat/completions` (vLLM, LM Studio, llama.cpp).

| Setting | Meaning |
|---|---|
| `LOCAL_LLM_URL` | Base URL including `/v1`, for example `http://localhost:8000/v1` |
| `LOCAL_LLM_MODEL` | The model id the server serves (for vLLM, the `--model` value). Falls back to `OLLAMA_MODEL` |
| `LOCAL_LLM_API_KEY` | Sent as `Authorization: Bearer ...`. A secret: never printed, never in `repr()` |
| `LOCAL_LLM_TEMPLATE_KWARGS` | JSON merged in as `chat_template_kwargs`. The default `{"enable_thinking": false}` turns off Qwen3's thinking, which vLLM leaves **on**. Set it empty for a server that rejects unknown fields |

Any `<think>` block left in the text is removed (so is one that never closed), and a separate `reasoning` field is
ignored: the graders and the reader see only the answer. `make status` probes `GET /v1/models` and says whether the server
is unreachable, refused the key, or does not serve the model. If the server fails during a question, the answer falls
back to the labelled extractive one, as with Ollama. Tested against a stub that speaks the vLLM API over a Unix socket
(`tests/test_openai_server.py`).

`make scale-up SERVICE=vllm` starts the official image with `VLLM_MODEL` (default `Qwen/Qwen3-8B`) and `VLLM_API_KEY`
(empty: no authentication, loopback only). It needs an NVIDIA GPU and the NVIDIA Container Toolkit, so it cannot run on a
Mac, and **that profile has only had its configuration validated, never run**. The first start downloads the model from
Hugging Face. Unlike the ONNX models it is not SHA-256 pinned, and the host is not air-gapped until the cache volume is
filled and `HF_HUB_OFFLINE=1` is set for the container.

## Shared semantic cache

`CACHE_BACKEND=postgres` keeps the cache table in PostgreSQL instead of a local SQLite file, so several UI processes read
and write one cache. Set `CACHE_DB_URI` to a database (it can be the pgvector database or any other).

* **Sealing is unchanged.** Question, answer, citations and query vector are AES-GCM sealed with the row identity as
  additional data, so the database operator learns none of them and a row moved to another clearance fails to open. The
  columns that are not sealed are the same as in SQLite: workspace, clearance, index version, review flag, approver, time.
* **Every process needs the same data key and the same access key.** On one host, copy the key files or point
  `DATA_KEY_PATH` / `ACCESS_KEY_PATH` at shared ones. On separate hosts `KEY_STORE=auto` would create a *different* key
  per host (each OS keychain is local), so cache rows would silently never open and session tokens would not verify across
  hosts: use `KEY_STORE=file` with one distributed key file. Envelope encryption through a KMS is an interface and a stub only.
* **L1 is off.** The in-process answer cache cannot be evicted by another process's prune or delete, so every lookup
  reads the shared table. Semantic search still uses the in-memory vector index, re-synced with the table on every lookup.
* **The index version is part of the key.** It is a hash of the workspace's manifest in `bm25_indices/`, so processes
  that answer the same workspace must hold the same manifest (next section).

## What does not scale

The options above remove the vector store, the chat model and the cache from the list of single-host components. These
remain **local files**, and a deployment over several hosts must put each on shared storage or keep one writer:

* the keyword index (`bm25_indices/`, SQLite FTS5) and the manifests beside it, which the cache version depends on;
* `uploads/` and `knowledge_base/`, the source documents;
* the audit log (`data/audit.db`, a SQLite hash chain), its anchors, and the access policy (`access_policy.json`);
* the keys.

Several UI **processes on one host** (behind a reverse proxy, for example) work with all of these as they are: SQLite is
safe across processes on one machine. SQLite over a network file system is not, which is the case the shared cache removes
for the cache alone. Checkpoints can already be PostgreSQL (`DB_URI`, `make up`).

`make backup` covers the audit log, anchors, checkpoints and policy of this host. It does **not** back up Qdrant, pgvector
or a shared cache database; those are yours to back up (`docker volume` snapshots, `pg_dump`), and any of them can be rebuilt
from the documents with `make sync FORCE=1`.

## Compose profiles

| Profile | Image | Port (loopback) | Volume | Authentication |
|---|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.20.0` | `QDRANT_PORT` (6333) | `qdrant_data` | `QDRANT_API_KEY`, empty = none |
| `pgvector` | `pgvector/pgvector:pg16` | `PGVECTOR_PORT` (5433) | `pgvector_data` | `POSTGRES_PASSWORD` from `.env` |
| `vllm` | `vllm/vllm-openai:latest` | `VLLM_PORT` (8000) | `vllm_cache` | `VLLM_API_KEY`, empty = none |
| `caddy` | `caddy:2-alpine` | `RAG_HTTPS_PORT` (8443) | `caddy_data` | see [lan-tls.md](lan-tls.md) |

`pgvector/pgvector:pg16` and `vllm/vllm-openai:latest` are moving tags: pin a digest for a deployment you keep. Nothing
here is exposed beyond loopback by default; put a TLS proxy or a private network in front before reaching a service from
another host.
