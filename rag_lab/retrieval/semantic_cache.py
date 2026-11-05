"""Persistent, clearance-partitioned semantic and exact cache.

Released answers are stored in a persistent SQLite database (CACHE_DB_PATH) with a thread-safe,
bounded LRU in-memory L1 cache. Cached answers are bound to workspace, clearance level, index
manifest version, and query to prevent stale answers after re-indexing or unauthorized
cross-clearance leakage. Query, answer, citations, and query vector are sealed at rest
(AES-256-GCM, vault.py) with the row identity as additional authenticated data; rows that are not
sealed, or that were moved between rows, are ignored. Each entry records whether a person approved it
or an auto_approve account did, so a team that requires human review is never served an answer
nobody reviewed.

Two-stage retrieval:
1. Exact Match: Sub-millisecond lookup across the caller's authorized clearance
   hierarchy (from caller's clearance down to public).
2. Semantic Match: In-process vector cosine similarity against cached questions
   in the same workspace and index version clearing the similarity threshold. The two questions must
   also carry the same numbers and negations, so "VAT in 2024" never answers "VAT in 2025".
"""
from __future__ import annotations

import array
import collections
import collections.abc
import re
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

from rag_lab.security.access import allowed_clearances as get_allowed_clearances
from rag_lab.config import settings
from rag_lab.lang import NEGATIONS, detect_language, fold, words
from rag_lab.security.vault import InvalidTag, open_bytes, open_text, seal_bytes, seal_text

logger = logging.getLogger("semantic_cache")

MAX_L1_SIZE = 2048
# Semantic matching scans in Python, so it only looks at the newest entries of a workspace.
MAX_SEMANTIC_CANDIDATES = 2000
_L1_LOCK = threading.Lock()
_L1_GENERATIONS: "collections.OrderedDict[str, str]" = collections.OrderedDict()
_L1_CITATIONS: dict[str, list[str]] = {}
_L1_REVIEWED: dict[str, bool] = {}

_COLUMNS = (
    "key, workspace_id, clearance, index_version, query, generation, citations_json, "
    "query_vector, reviewed, approved_by, created_at"
)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
# Negations in every supported language (rag_lab.lang): "nicht", "pas", "δεν" change a question as much as "not".


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_version(ws_id: str) -> str:
    """Fingerprint of a workspace's indexed content and of the model that embeds questions now.

    Changes whenever the workspace is re-indexed, and whenever the embedding model changes: two models
    can share a vector size (bge-small and MiniLM both use 384), so old query vectors would otherwise be
    compared with new ones without any error.
    """
    from rag_lab.retrieval.embedding_provider import embedding_fingerprint

    manifest = Path(settings.bm25_dir) / f"{ws_id}_manifest.json"
    try:
        model = embedding_fingerprint()
    except Exception:
        model = "unknown"
    if not manifest.is_file():
        return f"none:{model}"
    try:
        return hashlib.sha256(manifest.read_bytes() + model.encode("utf-8")).hexdigest()
    except OSError:
        return f"none:{model}"


def cache_key(ws_id: str, query: str, clearance: str = "public") -> str:
    """Canonical exact cache key binding workspace, clearance, index version, and normalized query."""
    return _sha256(f"{ws_id}:{clearance}:{_index_version(ws_id)}:{query.strip().lower()}")


def _signature(text: str) -> tuple[frozenset[str], frozenset[str], str]:
    """The numbers, negations, and language of a question.

    Two questions that differ in any of them are not the same question: "VAT in 2024" never answers
    "VAT in 2025", "nicht erlaubt" never answers "erlaubt", and an answer cached for a German question is
    not served to the same question asked in English.
    """
    lowered = fold(text).replace("'", "").replace("\u2019", "")
    numbers = frozenset(m.group(0) for m in _NUMBER.finditer(lowered))
    return numbers, frozenset(words(lowered)) & NEGATIONS, detect_language(text)


def _pack_vector(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack_vector(blob: bytes) -> list[float]:
    if not blob or len(blob) % 4 != 0:
        return []
    try:
        arr = array.array("f")
        arr.frombytes(blob)
        return arr.tolist()
    except (ValueError, TypeError):
        return []


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# --- Sealing -----------------------------------------------------------------

def _aad(key: str, workspace_id: str, clearance: str, index_version: str) -> bytes:
    return f"cache|{key}|{workspace_id}|{clearance}|{index_version}".encode("utf-8")


def _seal_row(key: str, workspace_id: str, clearance: str, version: str, query: str, generation: str,
              citations: list[str], vector: bytes | None) -> tuple[str, str, str, bytes | None]:
    aad = _aad(key, workspace_id, clearance, version)
    return (
        seal_text(query, aad),
        seal_text(generation, aad),
        seal_text(json.dumps(citations), aad),
        seal_bytes(vector, aad) if vector else None,
    )


def _open_row(row: tuple) -> dict[str, Any] | None:
    """Decrypt one cache row. Returns None for rows that are unsealed, moved, or otherwise not trustworthy."""
    key, workspace_id, clearance, version, query, generation, cites, vector, reviewed, approved_by, created = row
    aad = _aad(key, workspace_id, clearance, version)
    try:
        return {
            "key": key,
            "workspace_id": workspace_id,
            "clearance": clearance,
            "query": open_text(query, aad, allow_plaintext=False),
            "generation": open_text(generation, aad, allow_plaintext=False),
            "citations": json.loads(open_text(cites, aad, allow_plaintext=False)),
            "vector": _unpack_vector(open_bytes(vector, aad, allow_plaintext=False)) if vector else [],
            "reviewed": bool(reviewed),
            "approved_by": approved_by or "",
            "created_at": created,
        }
    except (InvalidTag, ValueError, TypeError):
        logger.debug("Ignoring cache row %s: it failed to open", key)
        return None


@contextmanager
def _open_db(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager guaranteeing connection termination and WAL-mode configuration."""
    target = db_path or settings.cache_db_path
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                key TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                clearance TEXT NOT NULL,
                index_version TEXT NOT NULL,
                query TEXT NOT NULL,
                generation TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                query_vector BLOB,
                created_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(semantic_cache);")}
        if "reviewed" not in columns:
            conn.execute("ALTER TABLE semantic_cache ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 1;")
        if "approved_by" not in columns:
            conn.execute("ALTER TABLE semantic_cache ADD COLUMN approved_by TEXT NOT NULL DEFAULT '';")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_ws_clr_ver
            ON semantic_cache (workspace_id, clearance, index_version);
            """
        )
        yield conn
    finally:
        conn.close()


def _retention_cutoff() -> str:
    """Entries older than this are expired on read, not only when `make retention` runs."""
    return (datetime.now(UTC) - timedelta(days=settings.retention_days)).isoformat(timespec="seconds")


# --- L1 (LRU) ----------------------------------------------------------------

def _l1_put(key: str, generation: str, citations: list[str], reviewed: bool = True) -> None:
    with _L1_LOCK:
        if key in _L1_GENERATIONS:
            _L1_GENERATIONS.move_to_end(key)
        elif len(_L1_GENERATIONS) >= MAX_L1_SIZE:
            oldest = next(iter(_L1_GENERATIONS))
            _L1_GENERATIONS.pop(oldest, None)
            _L1_CITATIONS.pop(oldest, None)
            _L1_REVIEWED.pop(oldest, None)
        _L1_GENERATIONS[key] = generation
        _L1_CITATIONS[key] = citations
        _L1_REVIEWED[key] = reviewed


def _l1_get(key: str) -> tuple[bool, str, list[str]]:
    with _L1_LOCK:
        if key in _L1_GENERATIONS:
            _L1_GENERATIONS.move_to_end(key)
            return True, _L1_GENERATIONS[key], _L1_CITATIONS.get(key, [])
    return False, "", []


def _l1_is_reviewed(key: str) -> bool:
    with _L1_LOCK:
        return _L1_REVIEWED.get(key, True)


def _l1_evict(key: str) -> None:
    with _L1_LOCK:
        _L1_GENERATIONS.pop(key, None)
        _L1_CITATIONS.pop(key, None)
        _L1_REVIEWED.pop(key, None)


def _l1_clear() -> None:
    with _L1_LOCK:
        _L1_GENERATIONS.clear()
        _L1_CITATIONS.clear()
        _L1_REVIEWED.clear()


# --- Store and lookup --------------------------------------------------------

def store_cache(
    workspace_id: str,
    query: str,
    clearance: str,
    generation: str,
    citations: list[str] | None = None,
    key: str | None = None,
    db_path: str | None = None,
    compute_vector: bool = True,
    reviewed: bool = True,
    approved_by: str = "",
) -> str:
    """Persist a released answer into SQLite and L1 memory with optional query embedding vector.

    `reviewed` is True when a person approved the answer and False when an auto_approve account did.
    """
    cites = list(citations or [])
    ver = _index_version(workspace_id)
    ckey = key or cache_key(workspace_id, query, clearance)

    _l1_put(ckey, generation, cites, reviewed)

    vector: bytes | None = None
    if compute_vector and query.strip():
        try:
            from rag_lab.retrieval.embedding_provider import get_embedding_model

            vec = get_embedding_model().embed_query(query.strip())
            if vec:
                vector = _pack_vector(vec)
        except Exception as exc:
            logger.debug("Failed computing query vector for cache: %s", exc)

    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        sealed_query, sealed_gen, sealed_cites, sealed_vec = _seal_row(
            ckey, workspace_id, clearance, ver, query.strip(), generation, cites, vector
        )
        with _open_db(db_path) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO semantic_cache ({_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (ckey, workspace_id, clearance, ver, sealed_query, sealed_gen, sealed_cites, sealed_vec,
                 1 if reviewed else 0, approved_by, now),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Could not persist cache entry %s: %s", ckey, exc)

    return ckey


def _hit(entry: dict[str, Any], hit_type: str, similarity: float) -> dict[str, Any]:
    return {
        "hit": True,
        "hit_type": hit_type,
        "similarity": similarity,
        "key": entry["key"],
        "generation": entry["generation"],
        "citations": entry["citations"],
        "query": entry["query"],
        "reviewed": entry["reviewed"],
        "approved_by": entry["approved_by"],
    }


def lookup_cache(
    workspace_id: str,
    query: str,
    clearance: str = "public",
    allowed_clearances: list[str] | None = None,
    threshold: float | None = None,
    db_path: str | None = None,
    include_unreviewed: bool = True,
) -> dict[str, Any]:
    """Two-tier lookup: hierarchical exact match, then vector cosine similarity.

    With include_unreviewed=False, entries that an auto_approve account released (rather than a
    person) are skipped, so a caller whose team requires review never gets an unreviewed answer.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"hit": False}

    authorized = allowed_clearances if allowed_clearances is not None else get_allowed_clearances(clearance)
    if not authorized:
        return {"hit": False}

    sim_threshold = threshold if threshold is not None else settings.semantic_cache_similarity_threshold
    cutoff = _retention_cutoff()
    curr_version = _index_version(workspace_id)

    # 1. Hierarchical exact match across the caller's authorized clearance levels (highest to lowest)
    exact_keys = {clr: cache_key(workspace_id, cleaned_query, clr) for clr in reversed(authorized)}
    for candidate_key in exact_keys.values():
        hit, gen, cites = _l1_get(candidate_key)
        if hit and (include_unreviewed or _l1_is_reviewed(candidate_key)):
            return {
                "hit": True, "hit_type": "exact", "similarity": 1.0, "key": candidate_key,
                "generation": gen, "citations": cites, "query": cleaned_query,
                "reviewed": _l1_is_reviewed(candidate_key), "approved_by": "",
            }

    try:
        with _open_db(db_path) as conn:
            for clr, candidate_key in exact_keys.items():
                row = conn.execute(
                    f"SELECT {_COLUMNS} FROM semantic_cache WHERE key = ? AND created_at >= ?;",
                    (candidate_key, cutoff),
                ).fetchone()
                entry = _open_row(row) if row else None
                if entry and entry["clearance"] == clr and (include_unreviewed or entry["reviewed"]):
                    _l1_put(candidate_key, entry["generation"], entry["citations"], entry["reviewed"])
                    return _hit(entry, "exact", 1.0)

            # 2. Semantic vector similarity across the matching workspace and index version
            placeholders = ", ".join("?" for _ in authorized)
            rows = conn.execute(
                f"""
                SELECT {_COLUMNS} FROM semantic_cache
                WHERE workspace_id = ? AND index_version = ? AND clearance IN ({placeholders})
                  AND query_vector IS NOT NULL AND created_at >= ?
                ORDER BY created_at DESC LIMIT ?;
                """,
                [workspace_id, curr_version, *authorized, cutoff, MAX_SEMANTIC_CANDIDATES],
            ).fetchall()
    except Exception as exc:
        logger.debug("Cache lookup error: %s", exc)
        return {"hit": False}

    candidates = [e for e in (_open_row(r) for r in rows) if e and e["vector"]]
    if not include_unreviewed:
        candidates = [e for e in candidates if e["reviewed"]]
    if not candidates:
        return {"hit": False}

    try:
        from rag_lab.retrieval.embedding_provider import get_embedding_model

        q_vec = get_embedding_model().embed_query(cleaned_query)
        if not q_vec:
            return {"hit": False}
    except Exception as exc:
        logger.debug("Embedding generation for semantic cache lookup failed: %s", exc)
        return {"hit": False}

    query_signature = _signature(cleaned_query)
    best_entry: Optional[dict[str, Any]] = None
    best_sim = -1.0
    for entry in candidates:
        if _signature(entry["query"]) != query_signature:
            continue
        sim = _cosine_similarity(q_vec, entry["vector"])
        if sim > best_sim:
            best_sim, best_entry = sim, entry

    if best_entry is not None and best_sim >= sim_threshold:
        return _hit(best_entry, "semantic", round(float(best_sim), 4))
    return {"hit": False}


def _delete_where(sql_select: str, sql_delete: str, params: tuple, db_path: str | None, label: str) -> int:
    removed = 0
    try:
        with _open_db(db_path) as conn:
            for row in conn.execute(sql_select, params).fetchall():
                _l1_evict(row[0])
            removed = conn.execute(sql_delete, params).rowcount
            conn.commit()
    except Exception as exc:
        logger.debug("Error %s: %s", label, exc)
    return removed


def prune_stale_cache(workspace_id: str, db_path: str | None = None) -> int:
    """Evict cached entries whose index_version does not match current workspace manifest."""
    params = (workspace_id, _index_version(workspace_id))
    return _delete_where(
        "SELECT key FROM semantic_cache WHERE workspace_id = ? AND index_version != ?;",
        "DELETE FROM semantic_cache WHERE workspace_id = ? AND index_version != ?;",
        params, db_path, f"pruning stale cache for {workspace_id}",
    )


def prune_cache(retention_days: int | None = None, db_path: str | None = None, now: datetime | None = None) -> int:
    """Prune cache entries older than the retention threshold."""
    retention_days = settings.retention_days if retention_days is None else retention_days
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=retention_days)).isoformat(timespec="seconds")
    return _delete_where(
        "SELECT key FROM semantic_cache WHERE created_at < ?;",
        "DELETE FROM semantic_cache WHERE created_at < ?;",
        (cutoff,), db_path, "pruning cache by retention",
    )


def clear_workspace_cache(workspace_id: str, db_path: str | None = None) -> None:
    """Evict all cached entries belonging to a single workspace."""
    _delete_where(
        "SELECT key FROM semantic_cache WHERE workspace_id = ?;",
        "DELETE FROM semantic_cache WHERE workspace_id = ?;",
        (workspace_id,), db_path, f"clearing workspace cache for {workspace_id}",
    )


def clear_cache(db_path: str | None = None) -> None:
    """Wipe all cache entries from L1 memory and SQLite."""
    _l1_clear()
    try:
        with _open_db(db_path) as conn:
            conn.execute("DELETE FROM semantic_cache;")
            conn.commit()
    except Exception as exc:
        logger.debug("Error clearing cache: %s", exc)


def cache_stats(db_path: str | None = None) -> dict[str, Any]:
    """Retrieve cache entry counts, workspace distribution, and on-disk size."""
    target = db_path or settings.cache_db_path
    path = Path(target)
    if not path.is_file():
        return {"total_entries": 0, "workspaces": {}, "bytes": 0}

    total = 0
    workspaces: dict[str, int] = {}
    try:
        with _open_db(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM semantic_cache;").fetchone()
            total = row[0] if row else 0
            ws_rows = conn.execute(
                "SELECT workspace_id, COUNT(*) FROM semantic_cache GROUP BY workspace_id;"
            ).fetchall()
            workspaces = {r[0]: r[1] for r in ws_rows if r[0]}
    except Exception as exc:
        logger.debug("Error fetching cache stats: %s", exc)

    return {
        "total_entries": total,
        "workspaces": workspaces,
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _fetch_entry(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    row = conn.execute(f"SELECT {_COLUMNS} FROM semantic_cache WHERE key = ?;", (key,)).fetchone()
    return _open_row(row) if row else None


class PersistentSemanticCache(collections.abc.MutableMapping):
    """Dict-like proxy providing backward-compatible access to SEMANTIC_CACHE."""

    def __getitem__(self, key: str) -> Any:
        hit, gen, _ = _l1_get(key)
        if hit:
            return gen
        try:
            with _open_db() as conn:
                entry = _fetch_entry(conn, key)
                if entry:
                    _l1_put(key, entry["generation"], entry["citations"], entry["reviewed"])
                    return entry["generation"]
        except Exception:
            pass
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        _l1_put(key, str(value), [])
        try:
            sealed_query, sealed_gen, sealed_cites, _ = _seal_row(key, "", "public", "", "", str(value), [], None)
            with _open_db() as conn:
                now = datetime.now(UTC).isoformat(timespec="seconds")
                conn.execute(
                    f"INSERT OR REPLACE INTO semantic_cache ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (key, "", "public", "", sealed_query, sealed_gen, sealed_cites, None, 1, "", now),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("PersistentSemanticCache setitem error: %s", exc)

    def __delitem__(self, key: str) -> None:
        _l1_evict(key)
        try:
            with _open_db() as conn:
                conn.execute("DELETE FROM semantic_cache WHERE key = ?;", (key,))
                conn.commit()
        except Exception:
            pass

    def __iter__(self):
        try:
            with _open_db() as conn:
                rows = conn.execute("SELECT key FROM semantic_cache;").fetchall()
                with _L1_LOCK:
                    keys = set(_L1_GENERATIONS.keys()) | {r[0] for r in rows}
                return iter(keys)
        except Exception:
            with _L1_LOCK:
                return iter(set(_L1_GENERATIONS.keys()))

    def __len__(self) -> int:
        try:
            with _open_db() as conn:
                row = conn.execute("SELECT COUNT(*) FROM semantic_cache;").fetchone()
                return row[0] if row else len(_L1_GENERATIONS)
        except Exception:
            with _L1_LOCK:
                return len(_L1_GENERATIONS)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with _L1_LOCK:
            if key in _L1_GENERATIONS:
                return True
        try:
            with _open_db() as conn:
                row = conn.execute("SELECT 1 FROM semantic_cache WHERE key = ?;", (key,)).fetchone()
                return bool(row)
        except Exception:
            return False

    def clear(self) -> None:
        clear_cache()


class PersistentCitationsCache(collections.abc.MutableMapping):
    """Dict-like proxy providing backward-compatible access to SEMANTIC_CACHE_CITATIONS."""

    def __getitem__(self, key: str) -> list[str]:
        with _L1_LOCK:
            if key in _L1_CITATIONS:
                return _L1_CITATIONS[key]
        try:
            with _open_db() as conn:
                entry = _fetch_entry(conn, key)
                if entry and entry["citations"]:
                    with _L1_LOCK:
                        _L1_CITATIONS[key] = entry["citations"]
                    return list(entry["citations"])
        except Exception:
            pass
        raise KeyError(key)

    def _rewrite(self, key: str, cites: list[str]) -> None:
        with _open_db() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM semantic_cache WHERE key = ?;", (key,)).fetchone()
            entry = _open_row(row) if row else None
            if entry is None:
                return
            aad = _aad(key, row[1], row[2], row[3])
            conn.execute(
                "UPDATE semantic_cache SET citations_json = ? WHERE key = ?;",
                (seal_text(json.dumps(cites), aad), key),
            )
            conn.commit()

    def __setitem__(self, key: str, value: list[str]) -> None:
        cites = list(value)
        with _L1_LOCK:
            _L1_CITATIONS[key] = cites
        try:
            self._rewrite(key, cites)
        except Exception as exc:
            logger.debug("PersistentCitationsCache setitem error: %s", exc)

    def __delitem__(self, key: str) -> None:
        with _L1_LOCK:
            _L1_CITATIONS.pop(key, None)
        try:
            self._rewrite(key, [])
        except Exception:
            pass

    def __iter__(self):
        with _L1_LOCK:
            return iter(set(_L1_CITATIONS.keys()))

    def __len__(self) -> int:
        with _L1_LOCK:
            return len(_L1_CITATIONS)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with _L1_LOCK:
            return key in _L1_CITATIONS

    def clear(self) -> None:
        with _L1_LOCK:
            _L1_CITATIONS.clear()
