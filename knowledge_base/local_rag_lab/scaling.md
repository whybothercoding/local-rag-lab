# Scaling options

Nothing here is needed on one laptop. `make setup && make models && make ui` uses embedded Chroma, a SQLite semantic cache and Ollama or the in-process extractive answerer.
Every option below is off by default, starts nothing on its own, and can be turned off again by deleting one line of `.env`.

## Vector stores

Chroma is the default. Qdrant and pgvector are opt-in: `make scale-up SERVICE=qdrant` or `SERVICE=pgvector`, then set `VECTOR_STORE=qdrant` or `pgvector` and run `make sync ADMIN=<admin> FORCE=1`.
Switching does not migrate the old index.
At 20,000 chunks Qdrant answers queries about 3 times faster than embedded Chroma and publishes about 4 times faster. A median query takes 9.6 ms on Qdrant against 31.9 ms on Chroma.
pgvector lives inside PostgreSQL you may already run, uses an exact scan with no ANN index, and takes 12.0 ms per median query at 20,000 chunks.
Every store keeps the clearance guard, cosine similarity on Chroma's scale, atomic publishes and no unfiltered fallback.
Qdrant and pgvector store chunk text in the clear.

## Chat model: vLLM

`LLM_PROVIDER=openai` talks to any server that implements `/v1/chat/completions`, including vLLM, LM Studio and llama.cpp.
`make scale-up SERVICE=vllm` starts vLLM on a GPU host and needs an NVIDIA GPU. It serves `Qwen/Qwen3-8B` by default.
It gives throughput for many concurrent users and room for a bigger model.
Its model is not SHA-256 pinned, and that compose profile has only had its configuration validated, never run.
Ollama stays the default because it runs on a Mac, and the LLM is not the bottleneck for one user.

## Shared semantic cache

`CACHE_BACKEND=postgres` with `CACHE_DB_URI` lets several UI processes share one sealed cache. All of them must use the same data key.

## PostgreSQL

`make up` starts an optional PostgreSQL checkpointer in Docker on 127.0.0.1 and creates `.env` with a random password. `make down` stops it. `make db-password` gives an existing PostgreSQL volume the password in `.env`, once, after an upgrade.

## LAN TLS

`make lan-tls` puts a Caddy reverse proxy with a local CA in front of the UI on port 8443. `make ui LAN=1` exposes the UI to the network. See `docs/lan-tls.md`.

## What does not scale

This is a single-author lab project. Use a TLS proxy or a private network before reaching any service from another host.
Services bind to 127.0.0.1 by default.
