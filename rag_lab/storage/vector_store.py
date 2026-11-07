"""Where chunk vectors live: embedded Chroma by default, Qdrant or pgvector when a deployment outgrows one host.

Every store answers the same few questions per workspace (is it indexed, with which model, give me the nearest
chunks the caller may read, replace the whole index atomically), so the pipeline never learns which one it talks to.

Two rules hold for every adapter, and are enforced here rather than left to each of them:

* **The clearance guard is in the base class.** `VectorStore.query()` calls the adapter's `_search()` (which should
  filter on the server) and then applies `guard_hits()`: a chunk with no valid clearance tag, a clearance the caller
  may not read, or a workspace other than the one asked about is dropped. A buggy or compromised filter in an adapter
  therefore cannot leak a chunk, and a new adapter cannot forget the guard.
* **Similarity means cosine similarity on the same scale as Chroma's `1 - distance`.** The relevance gates are
  calibrated on that scale (make calibrate-models); a store with a different score scale would break them without any
  error, as the Ollama embedder did. tests/test_vector_stores.py asserts the values, not just the order.

Chroma is the default and its behaviour is unchanged: same collection names (`ws_<id>`), the same metadata, the same
atomic swap by renaming a `__build` collection. Qdrant and pgvector are optional (`VECTOR_STORE=qdrant|pgvector`,
docs/scaling.md).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from urllib.parse import urlsplit
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from rag_lab.config import settings
from rag_lab.retrieval.embedding_provider import FINGERPRINT_KEY

logger = logging.getLogger("vector_store")

STORES = ("chroma", "qdrant", "pgvector")
_TEMP_SUFFIXES = ("__build", "__old")


class VectorStoreError(RuntimeError):
    """The configured store is unknown, unreachable or not installed. The message says how to fix it."""


class WorkspaceNotIndexed(LookupError):
    """The store has no index for this workspace."""


@dataclass(frozen=True)
class Hit:
    id: str
    document: str
    metadata: dict[str, Any]
    similarity: float


def guard_hits(hits: Sequence[Hit], workspace: str, allowed_clearances: Sequence[str]) -> list[Hit]:
    """Fail-closed check applied to every hit of every adapter after the server-side filter.

    A chunk with no clearance tag is treated as unreadable, never as public.
    """
    allowed = {c.strip().lower() for c in allowed_clearances}
    kept: list[Hit] = []
    for hit in hits:
        meta = hit.metadata or {}
        clearance = str(meta.get("clearance", "")).strip().lower()
        if not clearance or clearance not in allowed:
            continue
        if meta.get("workspace", workspace) != workspace:
            continue
        kept.append(hit)
    return kept


def safe_location(uri: str) -> str:
    """host:port/path of a URL, without any user or password."""
    parts = urlsplit(uri)
    return f"{parts.hostname or uri}{':' + str(parts.port) if parts.port else ''}{parts.path}"


class VectorStore(ABC):
    name = ""

    def describe(self) -> str:
        """Where this store is, for status output. Never contains a credential."""
        return self.name

    # --- what an adapter implements -----------------------------------------------------------------
    @abstractmethod
    def workspaces(self) -> set[str]:
        """Workspaces that have an index (or were created empty), without the temporary copies of a re-index."""

    @abstractmethod
    def ensure(self, workspace: str) -> None:
        """Create an empty index for a new workspace if there is none."""

    @abstractmethod
    def exists(self, workspace: str) -> bool: ...

    @abstractmethod
    def count(self, workspace: str) -> int:
        """Chunks in the workspace's index. Raises WorkspaceNotIndexed when it has none."""

    @abstractmethod
    def collection_metadata(self, workspace: str) -> dict[str, Any] | None:
        """What the index records about itself: at least FINGERPRINT_KEY when known. Raises WorkspaceNotIndexed."""

    @abstractmethod
    def read_vectors(self, workspace: str) -> Iterator[tuple[str, list[float]]]:
        """(chunk text, vector) for every chunk, so an unchanged chunk need not be embedded again."""

    @abstractmethod
    def publish(
        self,
        workspace: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        vectors: list[list[float]],
        fingerprint: str,
    ) -> None:
        """Replace the workspace's index with these chunks atomically: readers see the old index or the new one, and a
        failure leaves the old one serving."""

    @abstractmethod
    def delete(self, workspace: str) -> None: ...

    @abstractmethod
    def _search(self, workspace: str, vector: list[float], n: int, allowed_clearances: Sequence[str]) -> list[Hit]:
        """Nearest chunks, best first, filtered on the server to the allowed clearances where the store can."""

    # --- what every store shares ---------------------------------------------------------------------
    def query(self, workspace: str, vector: list[float], n: int, allowed_clearances: Sequence[str]) -> list[Hit]:
        """Nearest chunks the caller may read. Raises WorkspaceNotIndexed when the workspace has no index."""
        return guard_hits(self._search(workspace, vector, n, allowed_clearances), workspace, allowed_clearances)


# --- Chroma (the default) --------------------------------------------------------------------------------------

def _similarity(distance: float, space: str) -> float:
    """Convert a Chroma distance to a cosine-like similarity for the collection's distance space.

    Collections built by this version use cosine distance. Older ones use squared L2 on unit vectors,
    where cosine = 1 - d / 2.
    """
    if space == "cosine":
        return 1.0 - distance
    return 1.0 - distance / 2.0


class ChromaStore(VectorStore):
    name = "chroma"
    BATCH = 256

    def describe(self) -> str:
        return f"embedded Chroma at {settings.chroma_path}"

    # Cosine distance keeps similarity meaningful whichever embedding model produced the vectors.
    METADATA = {"hnsw:space": "cosine"}

    def _client(self):
        import chromadb

        return chromadb.PersistentClient(path=settings.chroma_path)

    @staticmethod
    def _name(workspace: str) -> str:
        return f"ws_{workspace}"

    def _collection(self, workspace: str):
        from chromadb.errors import NotFoundError

        try:
            return self._client().get_collection(name=self._name(workspace))
        except NotFoundError as exc:
            raise WorkspaceNotIndexed(workspace) from exc

    def workspaces(self) -> set[str]:
        found = set()
        for col in self._client().list_collections():
            name = getattr(col, "name", str(col))
            if name.startswith("ws_") and not name.endswith(_TEMP_SUFFIXES):
                found.add(name[3:])
        return found

    def ensure(self, workspace: str) -> None:
        self._client().get_or_create_collection(name=self._name(workspace), metadata=dict(self.METADATA))

    def exists(self, workspace: str) -> bool:
        return self._name(workspace) in [getattr(c, "name", str(c)) for c in self._client().list_collections()]

    def count(self, workspace: str) -> int:
        return int(self._collection(workspace).count())

    def collection_metadata(self, workspace: str) -> dict[str, Any] | None:
        meta = getattr(self._collection(workspace), "metadata", None)
        return dict(meta) if isinstance(meta, dict) else None

    def read_vectors(self, workspace: str) -> Iterator[tuple[str, list[float]]]:
        collection = self._collection(workspace)
        total = collection.count()
        for offset in range(0, total, 1000):
            page = collection.get(include=["documents", "embeddings"], limit=1000, offset=offset)
            texts, vectors = page.get("documents") or [], page.get("embeddings")
            if vectors is None or len(texts) != len(vectors):
                raise VectorStoreError(f"the index of '{workspace}' returned documents without matching vectors")
            for text, vector in zip(texts, vectors):
                yield text, [float(x) for x in vector]

    def publish(self, workspace, ids, documents, metadatas, vectors, fingerprint) -> None:
        """Build the new collection beside the live one, then swap them by renaming.

        Readers see the old index until the swap, and the gap while it happens is two renames long. If the swap fails
        the old collection is put back, so a failed publish never loses the live index.
        """
        client = self._client()
        name = self._name(workspace)
        build_name, old_name = f"{name}__build", f"{name}__old"
        self._delete_quiet(client, build_name)
        self._delete_quiet(client, old_name)
        building = client.create_collection(name=build_name, metadata={**self.METADATA, FINGERPRINT_KEY: fingerprint})
        for start in range(0, len(ids), self.BATCH):
            end = start + self.BATCH
            building.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end],
                         embeddings=vectors[start:end])

        live_exists = name in [getattr(c, "name", str(c)) for c in client.list_collections()]
        if live_exists:
            client.get_collection(name).modify(name=old_name)
        try:
            building.modify(name=name)
        except Exception:
            logger.exception("Could not publish collection %s; restoring the previous index", name)
            if live_exists:
                client.get_collection(old_name).modify(name=name)
            raise
        self._delete_quiet(client, old_name)

    def delete(self, workspace: str) -> None:
        client = self._client()
        name = self._name(workspace)
        for target in (name, f"{name}__build", f"{name}__old"):
            self._delete_quiet(client, target)

    @staticmethod
    def _delete_quiet(client, name: str) -> None:
        try:
            client.delete_collection(name=name)
        except Exception as exc:
            logger.debug("Collection %s was not deleted (%s)", name, exc)

    def _search(self, workspace, vector, n, allowed_clearances) -> list[Hit]:
        collection = self._collection(workspace)
        raw_meta = getattr(collection, "metadata", None)
        space = str(raw_meta.get("hnsw:space", "l2")) if isinstance(raw_meta, dict) else "l2"
        try:
            results = collection.query(
                query_embeddings=[vector],  # type: ignore[arg-type]  # chroma stubs want numpy; lists are accepted
                n_results=n,
                where={"clearance": {"$in": list(allowed_clearances)}},  # type: ignore[dict-item]
            )
        except Exception as exc:
            # Kept for parity with the behaviour before this interface existed: ask without the filter and let the
            # guard in VectorStore.query() do the filtering. Other adapters fail closed instead. The failure is
            # logged (never silent) and the fetch is widened so the guard still finds `n` readable chunks.
            logger.warning("Chroma workspace filter failed for %s (%s); fetching unfiltered and guarding in Python",
                           workspace, exc)
            results = collection.query(query_embeddings=[vector], n_results=max(n * 10, 100))  # type: ignore[arg-type]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [2.0] * len(docs)
        return [
            Hit(id=cid, document=doc, metadata=dict(meta or {}), similarity=_similarity(float(distance), space))
            for doc, meta, cid, distance in zip(docs, metas, ids, distances)
        ]


def vector_store_status() -> tuple[bool, str]:
    """(healthy, one line) for `make status`: which store, where, and whether it answers."""
    try:
        store = get_vector_store()
    except VectorStoreError as exc:
        return False, f"{settings.vector_store}: {exc}"
    try:
        return True, f"{store.describe()}: {len(store.workspaces())} workspace(s)"
    except Exception as exc:
        return False, f"{store.describe()} is unreachable ({type(exc).__name__}): dense retrieval is unavailable"


# --- The factory ---------------------------------------------------------------------------------------------------

_INSTANCES: dict[tuple, VectorStore] = {}
_INSTANCES_LOCK = threading.Lock()


def _digest(secret: str) -> str:
    """Names a connection string or key in a dict key without containing it (tracebacks and test output print keys)."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


def get_vector_store() -> VectorStore:
    """The store selected by VECTOR_STORE, read when called so a settings override applies."""
    kind = settings.vector_store
    if kind == "chroma":
        return ChromaStore()
    if kind == "qdrant":
        key: tuple = ("qdrant", settings.qdrant_url, _digest(settings.qdrant_api_key), settings.qdrant_prefix)
        if key not in _INSTANCES:
            with _INSTANCES_LOCK:
                if key not in _INSTANCES:  # the UI is multi-threaded: two first calls must not build two clients
                    try:
                        from rag_lab.storage.qdrant_store import QdrantStore
                    except ImportError as exc:
                        raise VectorStoreError(
                            "VECTOR_STORE=qdrant needs the optional qdrant-client package: pip install 'qdrant-client>=1.9' "
                            "(or pip install -e '.[qdrant]')."
                        ) from exc
                    _INSTANCES[key] = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_prefix)
        return _INSTANCES[key]
    if kind == "pgvector":
        key = ("pgvector", _digest(settings.pgvector_uri), settings.pgvector_schema)
        if key not in _INSTANCES:
            with _INSTANCES_LOCK:
                if key not in _INSTANCES:
                    from rag_lab.storage.pgvector_store import PgvectorStore

                    _INSTANCES[key] = PgvectorStore(settings.pgvector_uri, settings.pgvector_schema)
        return _INSTANCES[key]
    raise VectorStoreError(f"VECTOR_STORE must be one of {', '.join(STORES)}; got '{kind}'.")
