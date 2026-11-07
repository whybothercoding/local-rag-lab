"""An in-memory vector index for the semantic cache: one unit-length float32 matrix per partition.

A partition is (cache database, workspace, clearance, index version). Matching never reads across partitions: a
lookup asks only for the clearances the caller may see, so a row the caller may not read is not merely filtered
out, it is not in any matrix that is searched.

The index is a *candidate generator*, never the authority:

* Every lookup re-syncs the partition with the database first, by `(key, rowid, created_at, reviewed)`. That is a
  cheap query that decrypts nothing; only rows that are new or replaced are opened (AES-GCM, with the row identity
  as additional authenticated data, exactly as before). Rows that fail to open are never indexed, and not retried
  until they change. So a row edited or moved on disk leaves the matrix on the next lookup, and writes made by
  another process are seen.
* The caller re-fetches and re-opens the winning row and recomputes its cosine from the stored vector before it
  returns a hit (semantic_cache.lookup_cache).

This replaces a scan that fetched up to 2,000 rows per lookup, opened four sealed fields of each and recomputed
each question's signature, then took a pure-Python cosine: 365 ms at 2,000 entries, linear and capped. A matrix
product over the same entries takes well under a millisecond, and the cap moves from 2,000 to 10,000 per partition.
Brute force is exact; an approximate index (hnswlib) was not worth its dependency or its recall loss at this size.
"""
from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Newest entries kept per partition. 10,000 x 384 float32 is 15 MB.
MAX_INDEXED_PER_PARTITION = 10_000
# Memory the indexes may hold in total; the least recently used partitions go first.
MAX_INDEX_BYTES = 256 * 1024 * 1024
_LOAD_CHUNK = 400


@dataclass
class Match:
    similarity: float
    created: str
    key: str
    clearance: str


class PartitionIndex:
    def __init__(self, clearance: str) -> None:
        self.clearance = clearance
        self.keys: list[str] = []
        self.rowids: list[int] = []
        self.created: list[str] = []
        self.reviewed: list[bool] = []
        self.signatures: list[Any] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._rejected: set[tuple[str, int, str]] = set()
        self._last: list[tuple[str, int, str, int]] | None = None

    @property
    def size(self) -> int:
        return len(self.keys)

    @property
    def nbytes(self) -> int:
        return int(self.matrix.nbytes)

    def sync(
        self,
        light_rows: list[tuple[str, int, str, int]],
        load: Callable[[list[str]], dict[str, tuple[np.ndarray, Any]]],
    ) -> None:
        """Bring the partition in line with `light_rows` = (key, rowid, created_at, reviewed) of its current rows.

        `load(keys)` opens those rows and returns {key: (vector, signature)} for the ones that are trustworthy.
        """
        if self._last is not None and light_rows == self._last:
            return  # nothing changed since the last sync: one comparison in C instead of a pass in Python
        wanted = {key: (rowid, created, bool(reviewed)) for key, rowid, created, reviewed in light_rows}
        keep: list[int] = []
        for i, key in enumerate(self.keys):
            current = wanted.get(key)
            if current is not None and current[0] == self.rowids[i] and current[1] == self.created[i]:
                keep.append(i)
                self.reviewed[i] = current[2]  # a flag changed in place is picked up without opening the row
        kept = {self.keys[i] for i in keep}
        fresh = [key for key, (rowid, created, _) in wanted.items()
                 if key not in kept and (key, rowid, created) not in self._rejected]
        loaded: dict[str, tuple[np.ndarray, Any]] = {}
        for start in range(0, len(fresh), _LOAD_CHUNK):
            loaded.update(load(fresh[start:start + _LOAD_CHUNK]))
        # The fingerprint is recorded only once the rows are loaded: a load that raised must not mark the
        # partition synced, or the failed rows are never retried (silent permanent cache misses).
        self._last = light_rows
        for key in fresh:
            if key not in loaded:
                self._rejected.add((key, wanted[key][0], wanted[key][1]))
        if len(keep) == len(self.keys) and not loaded:
            return

        dim = self.matrix.shape[1]
        vectors: list[np.ndarray] = []
        keys, rowids, created, reviewed, signatures = [], [], [], [], []
        for i in keep:
            keys.append(self.keys[i])
            rowids.append(self.rowids[i])
            created.append(self.created[i])
            reviewed.append(self.reviewed[i])
            signatures.append(self.signatures[i])
        for key, (vector, signature) in loaded.items():
            if dim == 0:
                dim = vector.shape[0]
            if vector.shape[0] != dim:
                continue  # a vector of another size comes from another model: it can never be compared
            rowid, stamp, flag = wanted[key]
            keys.append(key)
            rowids.append(rowid)
            created.append(stamp)
            reviewed.append(flag)
            signatures.append(signature)
            vectors.append(vector)
        parts = ([self.matrix[keep]] if keep else []) + vectors
        self.keys, self.rowids, self.created, self.reviewed, self.signatures = keys, rowids, created, reviewed, signatures
        self.matrix = np.vstack(parts) if parts else np.zeros((0, dim), dtype=np.float32)
        if len(self._rejected) > 4 * MAX_INDEXED_PER_PARTITION:
            self._rejected = {r for r in self._rejected if r[0] in wanted}

    def search(
        self, query: np.ndarray, threshold: float, signature: Any, include_unreviewed: bool, limit: int = 8
    ) -> list[Match]:
        """Entries at or above `threshold` whose numbers, negations and language match, best first."""
        if not self.keys or query.shape[0] != self.matrix.shape[1]:
            return []
        scores = self.matrix @ query
        found: list[Match] = []
        for i in np.nonzero(scores >= threshold)[0]:
            if self.signatures[i] != signature:
                continue
            if not include_unreviewed and not self.reviewed[i]:
                continue
            found.append(Match(float(scores[i]), self.created[i], self.keys[i], self.clearance))
        found.sort(key=lambda m: m.created, reverse=True)  # on a tie, the newer entry
        found.sort(key=lambda m: -m.similarity)  # stable: similarity first
        return found[:limit]


def unit(vector: Any) -> np.ndarray | None:
    """A float32 unit vector, or None for an empty or zero vector (which is similar to nothing)."""
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        return None
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return array / norm


# --- The registry ------------------------------------------------------------------------------------------

LOCK = threading.RLock()
_INDEXES: "collections.OrderedDict[tuple[str, str, str, str], PartitionIndex]" = collections.OrderedDict()


def partition(db: str, workspace: str, clearance: str, version: str) -> PartitionIndex:
    """The index for one partition (created empty on first use). Hold `lock()` while syncing or searching it."""
    key = (db, workspace, clearance, version)
    with LOCK:
        index = _INDEXES.get(key)
        if index is None:
            index = _INDEXES[key] = PartitionIndex(clearance)
        _INDEXES.move_to_end(key)
        return index


def enforce_budget() -> None:
    """Drop the least recently used partitions while the indexes hold more than MAX_INDEX_BYTES."""
    with LOCK:
        while len(_INDEXES) > 1 and sum(i.nbytes for i in _INDEXES.values()) > MAX_INDEX_BYTES:
            _INDEXES.popitem(last=False)


def forget(db: str | None = None) -> None:
    """Drop every partition (or those of one database)."""
    with LOCK:
        for key in [k for k in _INDEXES if db is None or k[0] == db]:
            del _INDEXES[key]


def stats() -> dict[str, int]:
    with LOCK:
        return {"partitions": len(_INDEXES), "entries": sum(i.size for i in _INDEXES.values()),
                "bytes": sum(i.nbytes for i in _INDEXES.values())}
