# The question-answering pipeline

Every question runs through a LangGraph state machine. The generated topology is `docs/architecture.mmd`, and `make graph` regenerates it. A test fails when it is stale.

## Steps in order

1. **guardrail**: verifies the signed session token, checks the user's query permission and screens the question for prompt injection. A failure ends the run.
2. **contextualize**: rewrites a follow-up question into a standalone question, using only released turns.
3. **cache_lookup**: looks for a sealed semantic cache hit. A hit ends the run.
4. **router**: sends small talk to the chitchat node and document questions to retrieval.
5. **retrieve**: hybrid search combining dense vectors, BM25 keywords and a reranker, filtered by the user's clearance.
6. **sanitize_context**: screens retrieved chunks for indirect prompt injection.
7. **grade_docs**: grades relevance of the chunks. A weak result sends the question to **rewrite** and retrieval again.
8. **generate**: writes the answer from the approved chunks.
9. **grade_hallucination**: checks that every figure and every sentence is supported by the sources. A failure regenerates the answer.
10. **grade_relevance**: checks the answer addresses the question. An irrelevant answer goes back to rewrite.
11. **human_approval**: releases the answer automatically or waits for a reviewer.

## Bounded retries

Rewrites spend `MAX_RETRIEVAL_RETRIES`. Regenerations spend `MAX_GENERATION_RETRIES`, with grader feedback added to the prompt.
When a budget runs out the pipeline takes the circuit-break path and the answer goes to human review instead of looping forever.

## Grounding checks

Figures stated in the answer but absent from the sources are rejected first.
Then every answer sentence must be entailed by the pinned NLI model (`groundedness_min_entailment`).
Premises are single source sentences, near sentence pairs and heading-titled sentences. A verbatim quote passes without asking the model.
If the NLI model is unavailable, the grader falls back, with a visible label, to word overlap plus the cross-encoder.

## Degradation

If dense search, the reranker, the accelerator or the LLM is unavailable, the system falls back deterministically with a labelled warning. It never fails silently.
If the LLM fails during a question, the answer falls back to the labelled extractive answer.

## Question state

`asked_question` is the user's redacted literal question, shown to reviewers and in the audit log.
`original_question` is the standalone form, used for the cache key, retrieval, generation and relevance grading.
`sanitized_question` is the expandable search query.
