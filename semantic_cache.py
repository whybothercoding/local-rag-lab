"""Persistent, clearance-partitioned semantic and exact cache.

Answers approved by human compliance are stored in a persistent SQLite database
(CACHE_DB_PATH) with a thread-safe, bounded in-memory L1 cache. Cached answers
are bound to workspace, clearance level, index manifest version, and query to prevent
stale answers after re-indexing or unauthorized cross-clearance leakage.

Two-stage retrieval:
1. Exact Match: Sub-millisecond lookup across the caller's authorized clearance
   hierarchy (from caller's clearance down to public).
2. Semantic Match: In-process vector cosine similarity against cached questions
   in the same workspace and index version clearing the similarity threshold.
"""
from __future__ import annotations

import array
import collections.abc
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, List, Optional

import workspace_manager
from access import allowed_clearances as get_allowed_clearances
from config import CACHE_DB_PATH, RETENTION_DAYS, SEMANTIC_CACHE_SIMILARITY_THRESHOLD

logger = logging.getLogger("semantic_cache")

MAX_L1_SIZE = 2048
_L1_LOCK = threading.Lock()
_L1_GENERATIONS: dict[str, str] = {}
_L1_CITATIONS: dict[str, list[str]] = {}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_version(ws_id: str) -> str:
    """Fingerprint of a workspace's indexed content. Changes whenever the workspace is re-indexed."""
    manifest = Path(workspace_manager.BM25_DIR) / f"{ws_id}_manifest.json"
    if not manifest.is_file():
        return "none"
    try:
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError:
        return "none"


def cache_key(ws_id: str, query: str, clearance: str = "public") -> str:
    """Canonical exact cache key binding workspace, clearance, index version, and normalized query."""
    return _sha256(f"{ws_id}:{clearance}:{_index_version(ws_id)}:{query.strip().lower()}")


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


@contextmanager
def _open_db(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager guaranteeing connection termination and WAL-mode configuration."""
    target = db_path or os.getenv("CACHE_DB_PATH", CACHE_DB_PATH)
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
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_ws_clr_ver
            ON semantic_cache (workspace_id, clearance, index_version);
            """
        )
        yield conn
    finally:
        conn.close()


def _l1_put(key: str, generation: str, citations: list[str]) -> None:
    with _L1_LOCK:
        if len(_L1_GENERATIONS) >= MAX_L1_SIZE and key not in _L1_GENERATIONS:
            oldest = next(iter(_L1_GENERATIONS))
            _L1_GENERATIONS.pop(oldest, None)
            _L1_CITATIONS.pop(oldest, None)
        _L1_GENERATIONS[key] = generation
        _L1_CITATIONS[key] = citations


def _l1_get(key: str) -> tuple[bool, str, list[str]]:
    with _L1_LOCK:
        if key in _L1_GENERATIONS:
            return True, _L1_GENERATIONS[key], _L1_CITATIONS.get(key, [])
    return False, "", []


def _l1_evict(key: str) -> None:
    with _L1_LOCK:
        _L1_GENERATIONS.pop(key, None)
        _L1_CITATIONS.pop(key, None)


def _l1_clear() -> None:
    with _L1_LOCK:
        _L1_GENERATIONS.clear()
        _L1_CITATIONS.clear()


def store_cache(
    workspace_id: str,
    query: str,
    clearance: str,
    generation: str,
    citations: list[str] | None = None,
    key: str | None = None,
    db_path: str | None = None,
    compute_vector: bool = True,
) -> str:
    """Persist an approved answer into SQLite and L1 memory with optional query embedding vector."""
    cites = list(citations or [])
    ver = _index_version(workspace_id)
    ckey = key or cache_key(workspace_id, query, clearance)

    _l1_put(ckey, generation, cites)

    q_blob: bytes | None = None
    if compute_vector and query.strip():
        try:
            from embedding_provider import get_embedding_model

            vec = get_embedding_model().embed_query(query.strip())
            if vec:
                q_blob = _pack_vector(vec)
        except Exception as exc:
            logger.debug("Failed computing query vector for cache: %s", exc)

    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with _open_db(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_cache (
                    key, workspace_id, clearance, index_version, query, generation, citations_json, query_vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    ckey,
                    workspace_id,
                    clearance,
                    ver,
                    query.strip(),
                    generation,
                    json.dumps(cites),
                    q_blob,
                    now,
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Could not persist cache entry %s: %s", ckey, exc)

    return ckey


def lookup_cache(
    workspace_id: str,
    query: str,
    clearance: str = "public",
    allowed_clearances: list[str] | None = None,
    threshold: float | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Two-tier lookup: hierarchical exact match (<0.2ms), then vector cosine similarity."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"hit": False}

    authorized = allowed_clearances if allowed_clearances is not None else get_allowed_clearances(clearance)
    if not authorized:
        return {"hit": False}

    sim_threshold = threshold if threshold is not None else SEMANTIC_CACHE_SIMILARITY_THRESHOLD

    # 1. Hierarchical exact match across caller's authorized clearance levels (highest to lowest)
    for clr in reversed(authorized):
        candidate_key = cache_key(workspace_id, cleaned_query, clr)

        hit, gen, cites = _l1_get(candidate_key)
        if hit:
            return {
                "hit": True,
                "hit_type": "exact",
                "similarity": 1.0,
                "key": candidate_key,
                "generation": gen,
                "citations": cites,
                "query": cleaned_query,
            }

        try:
            with _open_db(db_path) as conn:
                row = conn.execute(
                    "SELECT generation, citations_json, query FROM semantic_cache WHERE key = ?;",
                    (candidate_key,),
                ).fetchone()
                if row:
                    gen, cites_json, orig_q = row
                    cites = json.loads(cites_json) if cites_json else []
                    _l1_put(candidate_key, gen, cites)
                    return {
                        "hit": True,
                        "hit_type": "exact",
                        "similarity": 1.0,
                        "key": candidate_key,
                        "generation": gen,
                        "citations": cites,
                        "query": orig_q,
                    }
        except Exception as exc:
            logger.debug("Exact cache lookup error: %s", exc)

    # 2. Semantic vector similarity search across matching workspace and index version
    curr_version = _index_version(workspace_id)
    candidates: list[tuple[str, str, str, str, bytes]] = []
    try:
        with _open_db(db_path) as conn:
            placeholders = ", ".join("?" for _ in authorized)
            sql = f"""
                SELECT key, query, generation, citations_json, query_vector
                FROM semantic_cache
                WHERE workspace_id = ? AND index_version = ? AND clearance IN ({placeholders})
                  AND query_vector IS NOT NULL;
            """
            params = [workspace_id, curr_version] + list(authorized)
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                if r[4]:
                    candidates.append(r)
    except Exception as exc:
        logger.debug("Semantic cache candidate fetch error: %s", exc)
        return {"hit": False}

    if not candidates:
        return {"hit": False}

    try:
        from embedding_provider import get_embedding_model

        q_vec = get_embedding_model().embed_query(cleaned_query)
        if not q_vec:
            return {"hit": False}
    except Exception as exc:
        logger.debug("Embedding generation for semantic cache lookup failed: %s", exc)
        return {"hit": False}

    best_match: Optional[dict[str, Any]] = None
    best_sim = -1.0

    for cand_key, cand_query, cand_gen, cand_cites_json, cand_blob in candidates:
        cand_vec = _unpack_vector(cand_blob)
        if not cand_vec:
            continue
        sim = _cosine_similarity(q_vec, cand_vec)
        if sim > best_sim:
            best_sim = sim
            cites = json.loads(cand_cites_json) if cand_cites_json else []
            best_match = {
                "hit": True,
                "hit_type": "semantic",
                "similarity": round(float(sim), 4),
                "key": cand_key,
                "generation": cand_gen,
                "citations": cites,
                "query": cand_query,
            }

    if best_match and best_sim >= sim_threshold:
        return best_match

    return {"hit": False}


def prune_stale_cache(workspace_id: str, db_path: str | None = None) -> int:
    """Evict cached entries whose index_version does not match current workspace manifest."""
    curr_ver = _index_version(workspace_id)
    removed = 0
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM semantic_cache WHERE workspace_id = ? AND index_version != ?;",
                (workspace_id, curr_ver),
            ).fetchall()
            for r in rows:
                _l1_evict(r[0])
            cursor = conn.execute(
                "DELETE FROM semantic_cache WHERE workspace_id = ? AND index_version != ?;",
                (workspace_id, curr_ver),
            )
            removed = cursor.rowcount
            conn.commit()
    except Exception as exc:
        logger.debug("Error pruning stale cache for %s: %s", workspace_id, exc)
    return removed


def prune_cache(retention_days: int = RETENTION_DAYS, db_path: str | None = None, now: datetime | None = None) -> int:
    """Prune cache entries older than the retention threshold."""
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=retention_days)).isoformat(timespec="seconds")
    removed = 0
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute("SELECT key FROM semantic_cache WHERE created_at < ?;", (cutoff,)).fetchall()
            for r in rows:
                _l1_evict(r[0])
            cursor = conn.execute("DELETE FROM semantic_cache WHERE created_at < ?;", (cutoff,))
            removed = cursor.rowcount
            conn.commit()
    except Exception as exc:
        logger.debug("Error pruning cache by retention: %s", exc)
    return removed


def clear_workspace_cache(workspace_id: str, db_path: str | None = None) -> None:
    """Evict all cached entries belonging to a single workspace."""
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM semantic_cache WHERE workspace_id = ?;",
                (workspace_id,),
            ).fetchall()
            for r in rows:
                _l1_evict(r[0])
            conn.execute("DELETE FROM semantic_cache WHERE workspace_id = ?;", (workspace_id,))
            conn.commit()
    except Exception as exc:
        logger.debug("Error clearing workspace cache for %s: %s", workspace_id, exc)


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
    target = db_path or os.getenv("CACHE_DB_PATH", CACHE_DB_PATH)
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


class PersistentSemanticCache(collections.abc.MutableMapping):
    """Dict-like proxy providing backward-compatible access to SEMANTIC_CACHE."""

    def __getitem__(self, key: str) -> Any:
        hit, gen, _ = _l1_get(key)
        if hit:
            return gen
        try:
            with _open_db() as conn:
                row = conn.execute("SELECT generation FROM semantic_cache WHERE key = ?;", (key,)).fetchone()
                if row:
                    _l1_put(key, row[0], [])
                    return row[0]
        except Exception:
            pass
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        _l1_put(key, str(value), [])
        try:
            with _open_db() as conn:
                now = datetime.now(UTC).isoformat(timespec="seconds")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_cache (
                        key, workspace_id, clearance, index_version, query, generation, citations_json, query_vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (key, "", "public", "", "", str(value), "[]", None, now),
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
                row = conn.execute("SELECT citations_json FROM semantic_cache WHERE key = ?;", (key,)).fetchone()
                if row and row[0]:
                    cites = json.loads(row[0])
                    with _L1_LOCK:
                        _L1_CITATIONS[key] = cites
                    return cites
        except Exception:
            pass
        raise KeyError(key)

    def __setitem__(self, key: str, value: list[str]) -> None:
        cites = list(value)
        with _L1_LOCK:
            _L1_CITATIONS[key] = cites
        try:
            with _open_db() as conn:
                conn.execute(
                    "UPDATE semantic_cache SET citations_json = ? WHERE key = ?;",
                    (json.dumps(cites), key),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("PersistentCitationsCache setitem error: %s", exc)

    def __delitem__(self, key: str) -> None:
        with _L1_LOCK:
            _L1_CITATIONS.pop(key, None)
        try:
            with _open_db() as conn:
                conn.execute(
                    "UPDATE semantic_cache SET citations_json = '[]' WHERE key = ?;",
                    (key,),
                )
                conn.commit()
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
