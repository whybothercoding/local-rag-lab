# Local RAG Lab overview

Local RAG Lab is a local, offline retrieval-augmented generation pipeline built with LangGraph.
It checks every answer sentence against its sources with an NLI model.
It supports English, German, French and Greek.
It is a single-author lab project run on one machine, and it is not a deployed product.

## What it does

The pipeline is self-correcting: guardrail, cache, router, hybrid retrieval, relevance and groundedness grading, query rewriting and regeneration with bounded retries, then release or human review.
Hybrid retrieval combines dense vectors with SQLite FTS5 BM25 keyword search and a reranker.
Grounding works by sentence: figures absent from the sources are rejected first, then each answer sentence must be entailed by a source sentence.
The NLI model is pinned in full precision (fp32).

## Offline at runtime

The system makes no network calls at runtime, because `HF_HUB_OFFLINE=1` is set.
The network is used only by `make models`, which downloads the pinned models once.
Every model is pinned by SHA-256 in `model_integrity.py`.
The chat model is Ollama with `qwen3:8b` (thinking off), or any OpenAI-compatible server such as vLLM.
Without a chat model, a deterministic extractive answerer quotes the sources instead.

## Stack

Python 3.11 to 3.14, LangGraph, in-process ONNX embeddings, reranker and NLI models, ChromaDB as the default vector store, SQLite FTS5 for BM25, SQLite or PostgreSQL for state, and a Streamlit web UI.
An in-process knowledge graph supports GraphRAG retrieval.
An optional Rust accelerator in `crates/rag_accel` speeds up injection scanning, with a Python fallback.
The license is Apache-2.0.

## Quickstart

Run `make setup`, then `make models`, then `make add-user ACCOUNT=me ADMIN=1`, then `make sync ADMIN=me`, then `make ui ADMIN=me`.
The first account becomes the owner.
The UI binds to 127.0.0.1 on port 8501.
`make help` lists every command.
This workspace, `local_rag_lab`, is the knowledge base that ships with the project: it describes the project itself, so you can ask the system about its own features.
