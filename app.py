import re
import json
import time
import hashlib
from pathlib import Path
from typing import TypedDict, List, Dict, Any

import chromadb
from chromadb.errors import NotFoundError

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from config import (
    CHROMA_PATH,
    BM25_DIR,
    DB_URI,  # noqa: F401
    RELEVANCE_MIN_COVERAGE,
    RELEVANCE_MIN_SIMILARITY,
    GROUNDEDNESS_MIN_RATIO,
    GROUNDEDNESS_CROSS_ENCODER_THRESHOLD,
    RELEVANCE_CROSS_ENCODER_THRESHOLD,
    MAX_RETRIEVAL_RETRIES,
    MAX_GENERATION_RETRIES,
)
from access import (
    allowed_clearances,
    authorize,
    clearance_covers,
    get_user_clearance,
    verify_token,
)
from audit import record as audit_record
from embedding_provider import (
    EmbeddingUnavailableError,
    embedding_fingerprint,
    get_embedding_model,
    stored_fingerprint,
)
from llm_provider import DeterministicContextLLM, get_chat_llm, llm_is_fallback
from redaction import redact_pii
from context_sanitizer import sanitize_chunks, scan_chunk, scan_user_input
from reranker import rerank_with_status, score_pairs
from sparse_search import has_sparse_index, search_sparse_chunks
from semantic_cache import (
    PersistentCitationsCache,
    PersistentSemanticCache,
    cache_key,  # noqa: F401  (re-exported: the canonical key lives in semantic_cache)
    lookup_cache,
    store_cache,
)


class GraphState(TypedDict, total=False):
    workspace_id: str
    session_id: str
    principal_token: str
    principal: str
    question: str
    original_question: str
    sanitized_question: str
    documents: List[str]
    citations: List[str]
    generation: str
    intent: str
    cache_hit: bool
    is_safe: bool
    grade_status: str
    hallucination_status: str
    relevance_status: str
    # Retrieval retries (query rewrites). The name predates the split and is kept so saved threads still load.
    retry_count: int
    generation_retry_count: int
    # Why the last answer failed a grader, added to the next generation prompt. Fixed text plus extracted figures only.
    grader_feedback: str
    history: List[str]
    telemetry: Dict[str, Any]
    degraded: List[str]
    principal_clearance: str
    dropped_chunks: List[Dict[str, Any]]


StateInput = GraphState | dict[str, Any]
SEMANTIC_CACHE = PersistentSemanticCache()
SEMANTIC_CACHE_CITATIONS = PersistentCitationsCache()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ABSTAIN_MESSAGE = "I could not find relevant documentation in this workspace to answer your query."

STOPWORDS = frozenset({
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how", "does", "did", "have",
    "has", "had", "with", "from", "into", "about", "there", "their", "your", "this", "that", "these",
    "those", "must", "should", "would", "could", "under", "over", "between", "after", "before", "tell",
    "give", "show", "list", "they", "them", "were", "been", "being", "will", "shall", "each", "every",
    "some", "any", "details", "overview",
})
_WORD = re.compile(r"[a-z0-9]+")

# Retrieval problems that mean the evidence behind an answer may be incomplete or tampered with.
# An answer carrying one of these is never auto-approved: a person must look at it.
BLOCKING_DEGRADED_PREFIXES = (
    "security_warning",
    "dense_retrieval_unavailable",
    "sparse_retrieval_unavailable",
)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def _numbers(text: str, *, include_one: bool = True) -> set[str]:
    """Numbers in a text, with spelled-out ones folded to digits ("five" == "5", "1,000" == "1000")."""
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())
    found = {m.rstrip(".,:") for m in _NUMBER.findall(lowered)}
    found |= {
        _NUMBER_WORDS[w]
        for w in _WORD.findall(lowered)
        if w in _NUMBER_WORDS and (include_one or w != "one")
    }
    return found


def _unsupported_numbers(generation: str, documents: List[str], question: str) -> set[str]:
    """Numbers an answer states that neither the retrieved context nor the question contains."""
    stated = _numbers(_LIST_MARKER.sub("", generation), include_one=False)
    return stated - _numbers(" ".join(documents)) - _numbers(question)


def _keywords(text: str) -> list[str]:
    """Distinct content words in order, dropping filler. Used to turn a question into a search query."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(text.lower()):
        if word not in STOPWORDS and (len(word) >= 3 or word.isdigit() or re.fullmatch(r"[a-z]\d+", word)):
            seen.setdefault(word)
    return list(seen)


def _stem_variants(words: list[str]) -> list[str]:
    variants: dict[str, None] = {}
    for word in words:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                variants.setdefault(word[: -len(suffix)])
                break
    return [v for v in variants if v not in words]


def _content_terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) >= 4 and word not in STOPWORDS}


def _coverage(query_terms: set[str], chunk: str) -> float:
    """Share of the query's content words that appear in one chunk."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(_WORD.findall(chunk.lower()))) / len(query_terms)


def _similarity(distance: float, space: str) -> float:
    """Convert a Chroma distance to a cosine-like similarity for the collection's distance space.

    Collections built by this version use cosine distance. Older ones use squared L2 on unit vectors,
    where cosine = 1 - d / 2.
    """
    if space == "cosine":
        return 1.0 - distance
    return 1.0 - distance / 2.0


class _EmbeddingMismatch(RuntimeError):
    """Raised inside dense retrieval to skip it with a labelled note."""


def _manifest(ws_id: str) -> dict | None:
    """The workspace's ingestion manifest, or None. Used for indexes that predate fingerprinted collections."""
    path = Path(BM25_DIR) / f"{ws_id}_manifest.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _embedding_mismatch(ws_id: str, collection_metadata: Any) -> str | None:
    """A labelled note when the index was built by another embedding model than the runtime uses.

    Vectors from two models are not comparable (and usually differ in size), so dense retrieval is
    skipped instead of returning wrong neighbours or failing with an opaque dimension error.
    """
    built_with = stored_fingerprint(collection_metadata, _manifest(ws_id))
    runtime = embedding_fingerprint()
    if built_with is None or built_with == runtime:
        return None
    return (
        f"dense_retrieval_unavailable: the index was built with {built_with} but the runtime uses {runtime}; "
        "re-index with 'make sync FORCE=1' or restore EMBEDDING_PROVIDER"
    )


# --- HELPER: RECIPROCAL RANK FUSION (RRF) ---

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """Merges and re-ranks dense and sparse candidate chunks using Reciprocal Rank Fusion.
    Score = sum(1.0 / (k + rank_i))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:top_n]]


# --- NODES ---

def guardrail_input_node(state: StateInput) -> dict[str, Any]:
    # Callers must pass redacted input: the checkpoint stores the submitted state before this node runs.
    t0 = time.perf_counter()
    raw = state.get("question", "")
    ws_id = state.get("workspace_id", "default")
    session_id = state.get("session_id", "")
    # Identity comes only from a signed session token. A bare name in the input is ignored.
    principal = verify_token(str(state.get("principal_token", "")))
    telemetry = dict(state.get("telemetry", {}))

    # Authorization comes first: unauthorized callers get no processing and no retrieval.
    if not authorize(principal, ws_id, "query"):
        denied_query = redact_pii(raw)
        audit_record(
            "access_denied",
            workspace_id=ws_id,
            session_id=session_id,
            data={"principal": principal, "permission": "query", "query": denied_query},
        )
        return {
            "is_safe": False,
            "generation": "[ACCESS DENIED: this account is not authorized for the selected workspace.]",
            "sanitized_question": denied_query,
            "telemetry": telemetry,
        }

    # Same pattern set as retrieved documents, after Unicode and whitespace normalization.
    input_safe, injection_reason = scan_user_input(raw)
    if not input_safe:
        telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        safe_query = redact_pii(raw)
        audit_record(
            "query_received",
            workspace_id=ws_id,
            session_id=session_id,
            data={"status": "blocked", "reason": "prompt_injection", "pattern": injection_reason, "query": safe_query},
        )
        return {
            "is_safe": False,
            "generation": "[SECURITY VIOLATION: Adversarial prompt injection detected.]",
            "sanitized_question": safe_query,
            "telemetry": telemetry,
        }

    sanitized = redact_pii(raw)
    audit_record(
        "query_received",
        workspace_id=ws_id,
        session_id=session_id,
        data={"status": "accepted", "query": sanitized},
    )

    telemetry["guardrail_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "is_safe": True,
        "principal": principal,
        "principal_clearance": get_user_clearance(principal),
        "original_question": sanitized,
        "sanitized_question": sanitized,
        "telemetry": telemetry,
        # A conversation reuses its thread: every question starts with full retry budgets and no carried-over notes.
        "retry_count": 0,
        "generation_retry_count": 0,
        "grader_feedback": "",
        "degraded": [],
        "dropped_chunks": [],
    }


def cache_lookup_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
    clearance = get_user_clearance(principal) if principal else "public"
    telemetry = dict(state.get("telemetry", {}))

    # A caller who is not on auto_approve is only served answers a person approved.
    may_use_unreviewed = bool(principal and authorize(principal, ws_id, "auto_approve"))
    cache_res = lookup_cache(
        workspace_id=ws_id,
        query=query,
        clearance=clearance,
        include_unreviewed=may_use_unreviewed,
    )
    telemetry["cache_lookup_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if cache_res.get("hit"):
        telemetry["cache_hit"] = True
        telemetry["cache_hit_type"] = cache_res.get("hit_type", "exact")
        telemetry["cache_similarity"] = cache_res.get("similarity", 1.0)
        telemetry["cache_approved_by"] = cache_res.get("approved_by", "")
        cached = cache_res["generation"]
        cached_citations = cache_res.get("citations") or state.get("citations", [])
        audit_record(
            "cache_hit",
            workspace_id=ws_id,
            session_id=state.get("session_id", ""),
            data={
                "query": query,
                "hit_type": telemetry["cache_hit_type"],
                "similarity": telemetry["cache_similarity"],
                "approved_by": telemetry["cache_approved_by"],
                "answer_sha256": _sha256(str(cached)),
            },
        )
        return {
            "cache_hit": True,
            "generation": str(cached),
            "citations": cached_citations,
            "telemetry": telemetry,
        }

    telemetry["cache_hit"] = False
    return {"cache_hit": False, "telemetry": telemetry}


def router_node(state: StateInput) -> dict[str, Any]:
    raw_query = state.get("sanitized_question", "").lower()
    cleaned = " ".join(_WORD.findall(raw_query))
    words = cleaned.split()
    chitchat_phrases = ("who are you", "what can you do", "thank you")
    chitchat_words = {"hello", "hi", "hey", "thanks"}

    if any(phrase in cleaned for phrase in chitchat_phrases):
        return {"intent": "chitchat"}
    domain_terms = _content_terms(cleaned) - chitchat_words
    if any(w in chitchat_words for w in words) and not domain_terms:
        return {"intent": "chitchat"}
    return {"intent": "rag"}


def chitchat_node(state: StateInput) -> dict[str, Any]:
    audit_record(
        "chitchat",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={"query": state.get("sanitized_question", "")},
    )
    return {
        "generation": "Hello! I am your local enterprise knowledge assistant. Ask me anything grounded in your active workspace."
    }


def hybrid_retrieval_node(state: StateInput) -> dict[str, Any]:
    """Retrieve candidates, then keep them only if they are relevant enough to answer from.

    Relevance is judged two ways: the best cosine similarity from the vector store, and how many
    of the query's content words one chunk contains. If neither clears its threshold, no documents
    are passed on, and the graph rewrites the query and eventually abstains.
    """
    t0 = time.perf_counter()
    query = state.get("sanitized_question", "")
    ws_id = state.get("workspace_id", "default")
    telemetry = dict(state.get("telemetry", {}))
    degraded: List[str] = []

    dense_candidates: List[Dict[str, Any]] = []
    sparse_candidates: List[Dict[str, Any]] = []

    # 1. Resolve user clearance levels for Zero Trust document-level RBAC filtering
    principal = state.get("principal", "")
    if not principal:
        principal = verify_token(str(state.get("principal_token", "")))
    user_clr = get_user_clearance(principal) if principal else "public"
    authorized_clrs = allowed_clearances(user_clr)

    # 1. Dense retrieval (ChromaDB) with clearance filter + fail-closed post-query verification
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=f"ws_{ws_id}")
        raw_meta = getattr(collection, "metadata", None)
        space = str(raw_meta.get("hnsw:space", "l2")) if isinstance(raw_meta, dict) else "l2"
        mismatch = _embedding_mismatch(ws_id, raw_meta)
        if mismatch:
            raise _EmbeddingMismatch(mismatch)
        q_vec = get_embedding_model().embed_query(query)
        try:
            results = collection.query(
                query_embeddings=[q_vec], n_results=5, where={"clearance": {"$in": authorized_clrs}}
            )
        except Exception:
            results = collection.query(query_embeddings=[q_vec], n_results=20)
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [2.0] * len(docs)
        for doc, meta, cid, distance in zip(docs, metas, ids, distances):
            meta_dict = meta or {}
            # A chunk with no clearance tag is treated as unreadable, never as public.
            doc_clr = str(meta_dict.get("clearance", "")).strip().lower()
            if not doc_clr or doc_clr not in authorized_clrs:
                continue
            dense_candidates.append({
                "id": cid,
                "content": doc,
                "citation": f"[{cid}] {meta_dict.get('source', 'document')}",
                "similarity": _similarity(float(distance), space),
            })
    except NotFoundError:
        degraded.append("not_indexed: this workspace has no vector index yet")
    except _EmbeddingMismatch as exc:
        degraded.append(str(exc))
    except EmbeddingUnavailableError as exc:
        degraded.append(f"dense_retrieval_unavailable: {exc}")
    except Exception as exc:
        degraded.append(f"dense_retrieval_unavailable: {type(exc).__name__}")

    # 2. Sparse retrieval (SQLite FTS5 with BM25 ranking, filtered by clearance, legacy fallback)
    try:
        sparse_candidates = search_sparse_chunks(
            ws_id,
            query,
            top_n=5,
            allowed_clearances=authorized_clrs,
            base_dir=BM25_DIR,
        )
        if dense_candidates and not has_sparse_index(ws_id, base_dir=BM25_DIR):
            degraded.append("sparse_retrieval_unavailable: the keyword index is missing; re-index the workspace")
    except Exception as exc:
        degraded.append(f"sparse_retrieval_unavailable: {type(exc).__name__}: {exc}")

    # 3. Relevance gate
    candidates = dense_candidates + sparse_candidates
    best_similarity = max((c["similarity"] for c in dense_candidates), default=0.0)
    query_terms = _content_terms(query)
    best_coverage = max((_coverage(query_terms, c["content"]) for c in candidates), default=0.0)
    relevant = bool(candidates) and (
        best_similarity >= RELEVANCE_MIN_SIMILARITY or best_coverage >= RELEVANCE_MIN_COVERAGE
    )

    # 4. Fuse with Reciprocal Rank Fusion (RRF) only when the evidence is relevant, drop chunks that carry
    # injected instructions, and only then cut to the final top 3 with the cross-encoder. Screening before
    # the cut means a poisoned chunk is replaced by the next-best clean one instead of leaving a gap.
    fused_docs = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=8) if relevant else []
    clean_pool: List[Dict[str, Any]] = []
    dropped_chunks: List[Dict[str, Any]] = []
    for item in fused_docs:
        is_safe, reason = scan_chunk(item["content"])
        if is_safe:
            clean_pool.append(item)
        else:
            dropped_chunks.append({
                "citation": item["citation"],
                "reason": reason or "unknown_violation",
                "preview": item["content"][:120],
            })
    reranked_docs, rerank_note = rerank_with_status(query, clean_pool, top_n=3) if clean_pool else ([], None)
    if rerank_note:
        degraded.append(rerank_note)

    chunks_found = [item["content"] for item in reranked_docs]
    citations = [item["citation"] for item in reranked_docs]

    telemetry["retrieval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["dense_candidates_count"] = len(dense_candidates)
    telemetry["sparse_candidates_count"] = len(sparse_candidates)
    telemetry["fused_chunks_count"] = len(chunks_found)
    telemetry["best_similarity"] = round(best_similarity, 4)
    telemetry["best_coverage"] = round(best_coverage, 4)
    telemetry["relevant"] = relevant

    audit_record(
        "retrieval",
        workspace_id=ws_id,
        session_id=state.get("session_id", ""),
        data={
            "citations": citations,
            "dense_candidates": len(dense_candidates),
            "sparse_candidates": len(sparse_candidates),
            "best_similarity": telemetry["best_similarity"],
            "best_coverage": telemetry["best_coverage"],
            "relevant": relevant,
            "degraded": degraded,
            "retry_count": state.get("retry_count", 0),
        },
    )

    return {
        "documents": chunks_found,
        "citations": citations,
        "telemetry": telemetry,
        "degraded": degraded,
        "dropped_chunks": dropped_chunks,
    }


def sanitize_context_node(state: StateInput) -> dict[str, Any]:
    """Inspect retrieved chunks for indirect prompt injection and drop poisoned content."""
    t0 = time.perf_counter()
    raw_docs = state.get("documents", [])
    raw_citations = state.get("citations", [])
    telemetry = dict(state.get("telemetry", {}))
    degraded = list(state.get("degraded", []))
    ws_id = state.get("workspace_id", "default")
    session_id = state.get("session_id", "")

    # Chunks already screened out during retrieval are reported here together with anything this
    # second pass still finds, so the audit trail and the warning cover both.
    clean_docs, clean_citations, late_dropped = sanitize_chunks(raw_docs, raw_citations)
    dropped = list(state.get("dropped_chunks") or []) + late_dropped

    telemetry["context_sanitizer_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    telemetry["poisoned_chunks_dropped"] = len(dropped)

    if dropped:
        degraded.append(
            f"security_warning: dropped {len(dropped)} poisoned document chunk(s) containing indirect prompt injection payloads"
        )
        for violation in dropped:
            audit_record(
                "indirect_injection_detected",
                workspace_id=ws_id,
                session_id=session_id,
                data={
                    "citation": violation["citation"],
                    "reason": violation["reason"],
                    "preview": violation["preview"],
                },
            )

    return {
        "documents": clean_docs,
        "citations": clean_citations,
        "telemetry": telemetry,
        "degraded": degraded,
        "dropped_chunks": [],
    }


def grade_documents_node(state: StateInput) -> dict[str, Any]:
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if not docs:
        if retries >= MAX_RETRIEVAL_RETRIES:
            return {"grade_status": "circuit_break"}
        return {"grade_status": "rewrite", "retry_count": retries + 1}

    return {"grade_status": "approved"}


def _llm_keyword_query(question: str) -> str | None:
    """Ask a real model for a keyword search query. Returns None when only the extractive fallback is available."""
    try:
        llm = get_chat_llm()
        if llm_is_fallback(llm) or isinstance(llm, DeterministicContextLLM):
            return None
        prompt = (
            "Rewrite this question as one short keyword search query for a document index. "
            "Reply with the query only.\n\nQuestion: " + question
        )
        candidate = " ".join(str(llm.invoke(prompt).content).split())[:200]
        return candidate or None
    except Exception:
        return None


def query_rewrite_node(state: StateInput) -> dict[str, Any]:
    """Retry retrieval with a different query. Always derived from the user's original question.

    First retry: a model-written keyword query when a model is available, otherwise the question's
    content words without filler. Later retries add word-stem variants to widen the match.
    """
    base = state.get("original_question") or state.get("sanitized_question", "")
    attempt = int(state.get("retry_count", 0))
    words = _keywords(base)
    if attempt <= 1:
        rewritten = _llm_keyword_query(base) or " ".join(words)
    else:
        rewritten = " ".join(words + _stem_variants(words))
    # New documents are coming, so the previous answer's grader note no longer applies.
    return {"sanitized_question": rewritten or base, "grader_feedback": ""}


def _generation_prompt(context: str, question: str, feedback: str) -> str:
    """The answer prompt. Grader feedback goes after the instruction, so the extractive synthesizer, which
    parses the Context and Question sections, reads the same question either way."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based strictly on the context:"
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback} Write a corrected answer."
    return prompt


def generate_node(state: StateInput) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = state.get("documents", [])
    # A model answers the user's own question. The extractive synthesizer matches sentences by shared words,
    # so it is given the search query, which after a rewrite is the keyword form that found these documents.
    original_q = state.get("original_question") or state.get("sanitized_question", "")
    search_q = state.get("sanitized_question") or original_q
    telemetry = dict(state.get("telemetry", {}))
    # Keep retrieval degradations; drop fallback notes from an earlier attempt so they are not repeated.
    degraded = [d for d in state.get("degraded", []) if not d.startswith("generation_fallback")]

    if not docs:
        telemetry["abstained"] = True
        telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {"generation": ABSTAIN_MESSAGE, "telemetry": telemetry, "degraded": degraded}

    context = "\n\n---\n\n".join(docs)
    feedback = str(state.get("grader_feedback") or "")
    telemetry["abstained"] = False
    telemetry["generation_feedback_applied"] = bool(feedback)
    prompt = _generation_prompt(context, search_q, feedback)
    try:
        llm = get_chat_llm()
        if llm_is_fallback(llm):
            degraded.append("generation_fallback: the model server is unavailable; the answer is extractive, not model-written")
            telemetry["generation_fallback"] = True
        # The extractive synthesizer returns the same text for the same documents, so retrying it is pointless.
        deterministic = isinstance(llm, DeterministicContextLLM)
        telemetry["generator_deterministic"] = deterministic
        if not deterministic:
            prompt = _generation_prompt(context, original_q, feedback)
        response = llm.invoke(prompt)
        gen_content = str(response.content).strip()
    except Exception as exc:
        # The model failed. Answer extractively from the same context and flag it for the reviewer.
        gen_content = str(
            DeterministicContextLLM().invoke(_generation_prompt(context, search_q, feedback)).content
        ).strip()
        degraded.append(f"generation_fallback: {type(exc).__name__}; the answer is extractive, not model-written")
        telemetry["generation_fallback"] = True
        telemetry["generator_deterministic"] = True

    # Real-time outbound DLP filter
    gen_content = redact_pii(gen_content)
    telemetry["generation_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "generation": gen_content,
        "telemetry": telemetry,
        "degraded": degraded,
    }


_FEEDBACK_UNSUPPORTED = (
    "it was not supported by the context. State only what the context says, or say that the context "
    "does not contain the answer."
)
_FEEDBACK_EMPTY = "it was empty. Answer from the context, or say that the context does not contain the answer."


def _generation_failure(state: StateInput, telemetry: dict[str, Any], feedback: str) -> tuple[str, dict[str, Any]]:
    """Decide what a failed groundedness check does next: regenerate with feedback, or stop and escalate.

    A deterministic generator would return the same text, so it escalates at once instead of looping.
    """
    retries = int(state.get("generation_retry_count", 0))
    if telemetry.get("generator_deterministic"):
        telemetry["generation_retry_skipped"] = "deterministic generator; a retry would repeat the same answer"
        return "circuit_break", {}
    if retries >= MAX_GENERATION_RETRIES:
        return "circuit_break", {}
    return "retry", {"generation_retry_count": retries + 1, "grader_feedback": feedback}


def grade_hallucination_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    docs = state.get("documents", [])
    telemetry = dict(state.get("telemetry", {}))
    telemetry.pop("generation_retry_skipped", None)

    status = "grounded"
    update: dict[str, Any] = {"grader_feedback": ""}
    ratio: float | None = None
    semantic_score: float | None = None
    grader_method = "lexical_pass"

    if gen == ABSTAIN_MESSAGE or telemetry.get("abstained"):
        status = "grounded"
        grader_method = "abstain"
    elif not gen or not str(gen).strip():
        # Blank generation cannot be grounded
        grader_method = "empty_generation"
        status, update = _generation_failure(state, telemetry, _FEEDBACK_EMPTY)
    elif docs and (
        unsupported := _unsupported_numbers(
            gen, docs, state.get("original_question") or state.get("sanitized_question", "")
        )
    ):
        # A figure the sources do not contain is never grounded, however much other wording overlaps.
        grader_method = "numeric_mismatch"
        telemetry["unsupported_numbers"] = sorted(unsupported)
        feedback = (
            f"it stated figures that are not in the context ({', '.join(sorted(unsupported))}). "
            "Use only figures that appear in the context."
        )
        status, update = _generation_failure(state, telemetry, feedback)
    elif docs:
        doc_words = set(_WORD.findall(" ".join(docs).lower()))
        gen_words = [w for w in _WORD.findall(gen.lower()) if len(w) > 3]
        if gen_words:
            ratio = sum(1 for w in gen_words if w in doc_words) / len(gen_words)
            if ratio < GROUNDEDNESS_MIN_RATIO:
                # Fast lexical screening missed (<25% overlap). Fall back to pinned
                # in-process ONNX cross-encoder to detect semantically grounded paraphrases.
                scores = score_pairs(gen, docs[:5])
                if scores:
                    semantic_score = round(max(scores), 4)
                    if semantic_score >= GROUNDEDNESS_CROSS_ENCODER_THRESHOLD:
                        grader_method = "semantic_cross_encoder"
                    else:
                        grader_method = "semantic_rejected"
                        status, update = _generation_failure(state, telemetry, _FEEDBACK_UNSUPPORTED)
                else:
                    grader_method = "lexical_rejected"
                    status, update = _generation_failure(state, telemetry, _FEEDBACK_UNSUPPORTED)

    telemetry["groundedness_ratio"] = None if ratio is None else round(ratio, 4)
    telemetry["groundedness_grader"] = grader_method
    if semantic_score is not None:
        telemetry["semantic_groundedness_score"] = semantic_score

    audit_record(
        "groundedness",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "status": status,
            "grader": grader_method,
            "groundedness_ratio": telemetry["groundedness_ratio"],
            "semantic_score": semantic_score,
            "generation_retry_count": int(state.get("generation_retry_count", 0)),
            "retry_skipped": telemetry.get("generation_retry_skipped"),
        },
    )
    return {"hallucination_status": status, "telemetry": telemetry, **update}


def grade_answer_relevance_node(state: StateInput) -> dict[str, Any]:
    gen = state.get("generation", "")
    telemetry = dict(state.get("telemetry", {}))
    if gen == ABSTAIN_MESSAGE or telemetry.get("abstained"):
        telemetry["relevance_grader"] = "abstain"
        return {"relevance_status": "relevant", "telemetry": telemetry}

    if not gen or not str(gen).strip():
        # Blank generation cannot be a relevant answer
        telemetry["relevance_grader"] = "empty_generation"
        if state.get("retry_count", 0) >= MAX_RETRIEVAL_RETRIES:
            return {"relevance_status": "circuit_break", "telemetry": telemetry}
        return {"relevance_status": "irrelevant", "retry_count": state.get("retry_count", 0) + 1, "telemetry": telemetry}

    gen_lower = gen.lower()
    gen_words = set(_WORD.findall(gen_lower))
    q = state.get("original_question") or state.get("sanitized_question", "")
    retries = state.get("retry_count", 0)

    q_tokens = [t for t in _WORD.findall(q.lower()) if len(t) > 3 and t not in STOPWORDS]
    has_lexical_overlap = bool(not q_tokens or any(t in gen_words or t in gen_lower for t in q_tokens))

    status = "relevant"
    grader_method = "lexical_pass"
    semantic_score: float | None = None
    update: dict[str, Any] = {}

    if not has_lexical_overlap:
        # Fast lexical check missed (>3 char query tokens absent). Fall back to
        # pinned in-process ONNX cross-encoder to assess semantic relevance.
        scores = score_pairs(q, [gen])
        if scores:
            semantic_score = round(scores[0], 4)
            if semantic_score >= RELEVANCE_CROSS_ENCODER_THRESHOLD:
                grader_method = "semantic_cross_encoder"
            else:
                grader_method = "semantic_rejected"
                if retries >= MAX_RETRIEVAL_RETRIES:
                    status = "circuit_break"
                else:
                    status = "irrelevant"
                    update["retry_count"] = retries + 1
        else:
            grader_method = "lexical_rejected"
            if retries >= MAX_RETRIEVAL_RETRIES:
                status = "circuit_break"
            else:
                status = "irrelevant"
                update["retry_count"] = retries + 1

    telemetry["relevance_grader"] = grader_method
    if semantic_score is not None:
        telemetry["semantic_relevance_score"] = semantic_score

    audit_record(
        "answer_relevance",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "status": status,
            "grader": grader_method,
            "semantic_score": semantic_score,
            "retry_count": retries,
        },
    )
    return {"relevance_status": status, "telemetry": telemetry, **update}


def _normalize_decision(raw: Any) -> dict[str, str]:
    """Read the resume payload. The approver is the account a valid session token was issued to."""
    payload = raw if isinstance(raw, dict) else {"decision": raw}
    return {
        "decision": str(payload.get("decision", "")).strip().lower(),
        "approver": verify_token(str(payload.get("approver_token", ""))),
        "reason": redact_pii(str(payload.get("reason", "")).strip()),
    }


def _review_reasons(state: StateInput) -> list[str]:
    """Why an answer must be looked at by a person even for an account on auto_approve."""
    gen = state.get("generation", "")
    abstained = gen == ABSTAIN_MESSAGE or bool(state.get("telemetry", {}).get("abstained"))
    reasons: list[str] = []
    hallucination = state.get("hallucination_status", "") or "grounded"
    relevance = state.get("relevance_status", "") or "relevant"
    telemetry = state.get("telemetry", {}) or {}
    if hallucination != "grounded":
        reasons.append(f"groundedness check did not pass ({hallucination})")
        # Say why, so the reviewer knows what to look for without reading telemetry.
        if telemetry.get("unsupported_numbers"):
            reasons.append(
                "answer states figures not found in the sources: " + ", ".join(telemetry["unsupported_numbers"])
            )
        if telemetry.get("generation_retry_skipped"):
            reasons.append(f"not regenerated: {telemetry['generation_retry_skipped']}")
    if relevance != "relevant":
        reasons.append(f"relevance check did not pass ({relevance})")
    if not abstained:
        reasons.extend(
            note for note in state.get("degraded", []) if note.startswith(BLOCKING_DEGRADED_PREFIXES)
        )
    return reasons


def human_in_the_loop_node(state: StateInput) -> dict[str, Any]:
    # The payload is what approve.py displays. It carries the redacted question, never the raw input.
    display_q = state.get("original_question") or state.get("sanitized_question", "")
    gen = state.get("generation", "")
    history = list(state.get("history", []))
    ws_id = state.get("workspace_id", "default")
    principal = state.get("principal", "") or verify_token(str(state.get("principal_token", "")))
    requester_clearance = state.get("principal_clearance") or get_user_clearance(principal)
    abstained = gen == ABSTAIN_MESSAGE or bool(state.get("telemetry", {}).get("abstained"))

    # auto_approve skips the human gate only for answers that passed every automated check.
    reasons = _review_reasons(state)
    has_auto_approve = bool(principal and authorize(principal, ws_id, "auto_approve"))
    auto = has_auto_approve and not reasons
    if auto:
        review = {
            "decision": "approved",
            "approver": f"auto:{principal}",
            "reason": f"auto-approved via policy grant for {principal}",
        }
        approved = True
    else:
        review = _normalize_decision(interrupt({
            "action": "human_approval",
            "question": display_q,
            "citations": state.get("citations", []),
            "candidate_answer": state.get("generation", ""),
            "requested_by": principal,
            "requester_clearance": requester_clearance,
            "review_reasons": reasons,
        }))
        approved = review["decision"] == "approved"
        if approved and not review["approver"]:
            # An approval with no accountable reviewer is not released.
            approved = False
            review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: no approver identity"]))
        if approved and review["approver"] == state.get("principal", ""):
            # Separation of duties: the requester cannot approve their own query.
            approved = False
            review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: self-approval"]))
        if approved and not authorize(review["approver"], ws_id, "review"):
            approved = False
            review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: approver lacks review permission"]))
        if approved and not clearance_covers(get_user_clearance(review["approver"]), requester_clearance):
            # A reviewer reads the answer, so they must be cleared for what the requester may see.
            approved = False
            review["reason"] = "; ".join(filter(None, [review["reason"], "auto-rejected: approver clearance is below the requester's"]))

    # Written before the answer is released, so a failed write releases nothing.
    audit_record(
        "hitl_decision",
        workspace_id=state.get("workspace_id", "default"),
        session_id=state.get("session_id", ""),
        data={
            "decision": "approved" if approved else "rejected",
            "requested": review["decision"],
            "approver": review["approver"],
            "reason": review["reason"],
            "released": approved,
            "auto_approved": bool(auto and approved),
            "auto_approve_withheld": bool(has_auto_approve and reasons),
            "review_reasons": reasons,
            "answer_sha256": _sha256(gen),
            "citations": state.get("citations", []),
            "groundedness_ratio": state.get("telemetry", {}).get("groundedness_ratio"),
            "hallucination_status": state.get("hallucination_status", ""),
            "relevance_status": state.get("relevance_status", ""),
            "degraded": state.get("degraded", []),
        },
    )

    if approved:
        history.append(f"Assistant: {gen}")

        q = state.get("original_question") or state.get("sanitized_question", "")
        clearance = get_user_clearance(principal) if principal else "public"
        cites = list(state.get("citations", []))
        # An abstention is not an answer worth remembering: the next question may find the documents.
        if not abstained:
            store_cache(
                workspace_id=ws_id,
                query=q,
                clearance=clearance,
                generation=gen,
                citations=cites,
                reviewed=not auto,
                approved_by=review["approver"],
            )

        return {"generation": gen, "history": history}

    rejection_notice = "[RESPONSE REJECTED BY HUMAN COMPLIANCE]"
    history.append(f"Assistant: {rejection_notice}")
    return {"generation": rejection_notice, "citations": [], "degraded": [], "history": history}


# --- GRAPH EDGES & ROUTING ---

def route_safety(state: StateInput) -> str:
    return "pass" if state.get("is_safe", True) else "fail"

def route_cache(state: StateInput) -> str:
    return "hit" if state.get("cache_hit", False) else "miss"

def route_intent(state: StateInput) -> str:
    return state.get("intent", "rag")

def route_doc_grade(state: StateInput) -> str:
    status = state.get("grade_status", "approved")
    if status == "circuit_break":
        return "circuit_break"
    if status == "rewrite":
        return "rewrite"
    return "approved"

def route_hallucination(state: StateInput) -> str:
    status = state.get("hallucination_status", "grounded")
    if status == "circuit_break":
        return "circuit_break"
    if status == "retry":
        return "retry"
    return "grounded"

def route_relevance(state: StateInput) -> str:
    status = state.get("relevance_status", "relevant")
    if status == "circuit_break":
        return "circuit_break"
    if status == "irrelevant":
        return "irrelevant"
    return "relevant"


# --- BUILD STATE GRAPH ---

builder = StateGraph(GraphState)

builder.add_node("guardrail", guardrail_input_node)
builder.add_node("cache_lookup", cache_lookup_node)
builder.add_node("router", router_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("retrieve", hybrid_retrieval_node)
builder.add_node("sanitize_context", sanitize_context_node)
builder.add_node("grade_docs", grade_documents_node)
builder.add_node("rewrite", query_rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("grade_hallucination", grade_hallucination_node)
builder.add_node("grade_relevance", grade_answer_relevance_node)
builder.add_node("human_approval", human_in_the_loop_node)

builder.add_edge(START, "guardrail")

builder.add_conditional_edges(
    "guardrail",
    route_safety,
    {"pass": "cache_lookup", "fail": END}
)

builder.add_conditional_edges(
    "cache_lookup",
    route_cache,
    {"hit": END, "miss": "router"}
)

builder.add_conditional_edges(
    "router",
    route_intent,
    {"chitchat": "chitchat", "rag": "retrieve"}
)

builder.add_edge("chitchat", END)
builder.add_edge("retrieve", "sanitize_context")
builder.add_edge("sanitize_context", "grade_docs")

builder.add_conditional_edges(
    "grade_docs",
    route_doc_grade,
    {"approved": "generate", "rewrite": "rewrite", "circuit_break": "generate"}
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "grade_hallucination")

builder.add_conditional_edges(
    "grade_hallucination",
    route_hallucination,
    {"grounded": "grade_relevance", "retry": "generate", "circuit_break": "human_approval"}
)

builder.add_conditional_edges(
    "grade_relevance",
    route_relevance,
    {"relevant": "human_approval", "irrelevant": "rewrite", "circuit_break": "human_approval"}
)

builder.add_edge("human_approval", END)

workflow = builder
