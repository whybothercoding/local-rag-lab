# Production Compound AI System (LangGraph RAG)

A resilient, enterprise-grade Compound AI System implemented in LangGraph with local LLMs, hybrid search, self-evaluating gates, and PostgreSQL-backed Human-in-the-Loop persistence.

## Architecture

```
[ Input Guardrails & PII ]
          │
[ Semantic SHA-256 Cache ]
          │
[ Structured Router (Pydantic) ]
          │
[ Hybrid Retrieval (Dense Chroma + Sparse BM25 via RRF) ]
          │
[ CRAG Document Relevance Grader ] ──(Max Retries Exceeded)──> [ Fallback Circuit Breaker ]
          │ (Approved)
[ Attributed Generation ]
          │
[ Self-RAG Hallucination & Relevance Gates ]
          │ (Grounded & Useful)
[ HITL Breakpoint (interrupt) ] <───> [ PostgresSaver (Docker) ]
          │
[ Emitted Verified Output ]
```

## Prerequisites

- macOS / Linux
- Python 3.11+ (tested on Python 3.14)
- Docker & Docker Compose
- Ollama running locally with `qwen2.5-coder:7b` (`ollama run qwen2.5-coder:7b`)

## Quickstart

1. **Spin up persistence:**
   ```bash
   make up
   ```

2. **Run ingestion and query pipeline (pauses at approval gate):**
   ```bash
   make run
   ```

3. **Audit and resume via CLI operator:**
   ```bash
   make audit
   # Or headless:
   python approve.py --thread <thread_id> --action approved
   ```

4. **Run test suite:**
   ```bash
   make test
   ```

5. **Export architecture diagram:**
   ```bash
   make export
   open docs/architecture.png
   ```
