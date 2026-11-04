"""Qdrant as the vector store (VECTOR_STORE=qdrant). Optional: needs the `qdrant-client` package.

Layout. Each workspace has a *physical* collection `<prefix>ws_<id>__<token>` and an *alias* `<prefix>ws_<id>` that
points to the live one. Queries go through the alias. A publish builds a new physical collection beside the live one
and then moves the alias in a single request (delete + create is one atomic operation on the server), so readers see
the old index or the new one, and a publish that fails part-way never touches the alias. The model stamp (fingerprint)
is the collection's own metadata.

Scores. Qdrant's Cosine distance returns the cosine similarity as the score, the same scale as Chroma's
`1 - distance` that the relevance gates are calibrated on (tests/test_vector_stores.py asserts the values).

Differences from Chroma that matter: vectors are normalised on insert (direction is kept, which is all ranking uses), a
collection's vector size is fixed when it is created (so an empty workspace holds a one-dimensional placeholder until the
first publish), and there is no unfiltered fallback: a failing server-side filter makes dense retrieval unavailable
(labelled) instead of asking for everything and trusting the guard.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, cast

from qdrant_client import QdrantClient, models

from rag_lab.retrieval.embedding_provider import FINGERPRINT_KEY
from rag_lab.storage.vector_store import Hit, VectorStore, VectorStoreError, WorkspaceNotIndexed

logger = logging.getLogger("qdrant_store")

_BATCH = 256
_PLACEHOLDER = models.VectorParams(size=1, distance=models.Distance.COSINE)


class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(self, url: str, api_key: str = "", prefix: str = "") -> None:
        if not url:
            raise VectorStoreError("VECTOR_STORE=qdrant needs QDRANT_URL (the compose profile listens on http://localhost:6333).")
        self._prefix = prefix
        self._client = QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(
            url=url, api_key=api_key or None, timeout=30
        )

    # --- naming ------------------------------------------------------------------------------------------------
    def _alias(self, workspace: str) -> str:
        return f"{self._prefix}ws_{workspace}"

    def _aliases(self) -> dict[str, str]:
        """{alias: physical collection} for this install's workspaces only."""
        head = f"{self._prefix}ws_"
        return {a.alias_name: a.collection_name for a in self._client.get_aliases().aliases if a.alias_name.startswith(head)}

    def _live(self, workspace: str) -> str | None:
        return self._aliases().get(self._alias(workspace))

    def _require(self, workspace: str) -> str:
        live = self._live(workspace)
        if live is None:
            raise WorkspaceNotIndexed(workspace)
        return live

    # --- the interface -------------------------------------------------------------------------------------------
    def workspaces(self) -> set[str]:
        head = f"{self._prefix}ws_"
        return {alias[len(head):] for alias in self._aliases()}

    def exists(self, workspace: str) -> bool:
        return self._live(workspace) is not None

    def ensure(self, workspace: str) -> None:
        if self._live(workspace) is not None:
            return
        physical = f"{self._alias(workspace)}__{uuid.uuid4().hex[:12]}"
        self._client.create_collection(physical, vectors_config=_PLACEHOLDER, metadata={})
        self._set_alias(self._alias(workspace), physical, previous=None)

    def count(self, workspace: str) -> int:
        self._require(workspace)
        return int(self._client.count(self._alias(workspace), exact=True).count)

    def collection_metadata(self, workspace: str) -> dict[str, Any] | None:
        info = self._client.get_collection(self._require(workspace))
        return dict(getattr(info.config, "metadata", None) or {})

    def read_vectors(self, workspace: str) -> Iterator[tuple[str, list[float]]]:
        self._require(workspace)
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._alias(workspace), limit=500, offset=offset, with_payload=["document"], with_vectors=True
            )
            for point in points:
                vector = point.vector
                if not isinstance(vector, list) or not point.payload or "document" not in point.payload:
                    raise VectorStoreError(f"the index of '{workspace}' returned a chunk without its text or vector")
                yield str(point.payload["document"]), [float(x) for x in cast(list[float], vector)]
            if offset is None:
                return

    def publish(self, workspace, ids, documents, metadatas, vectors, fingerprint) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(vectors)):
            raise ValueError("ids, documents, metadatas and vectors must be the same length")
        if not ids:
            raise ValueError("nothing to publish")
        sizes = {len(v) for v in vectors}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(f"vectors must all have the same non-zero size; got {sorted(sizes)}")
        alias = self._alias(workspace)
        previous = self._aliases().get(alias)
        physical = f"{alias}__{uuid.uuid4().hex[:12]}"
        self._client.create_collection(
            physical,
            vectors_config=models.VectorParams(size=sizes.pop(), distance=models.Distance.COSINE),
            metadata={FINGERPRINT_KEY: fingerprint},
        )
        try:
            try:
                self._client.create_payload_index(physical, "clearance", models.PayloadSchemaType.KEYWORD)
            except Exception:  # the embedded mode has no payload indexes; the server filters correctly without one
                logger.debug("No payload index on clearance for %s", physical)
            for start in range(0, len(ids), _BATCH):
                end = start + _BATCH
                self._client.upsert(
                    physical,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                            vector=[float(x) for x in vector],
                            payload={**meta, "id": cid, "document": doc, "workspace": workspace},
                        )
                        for cid, doc, meta, vector in zip(ids[start:end], documents[start:end], metadatas[start:end], vectors[start:end])
                    ],
                    wait=True,
                )
            self._set_alias(alias, physical, previous)
        except BaseException:
            self._drop_collection(physical)
            raise
        if previous and previous != physical:
            self._drop_collection(previous)

    def delete(self, workspace: str) -> None:
        alias = self._alias(workspace)
        live = self._aliases().get(alias)
        if live is not None:
            try:
                self._client.update_collection_aliases(
                    change_aliases_operations=[models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))]
                )
            except Exception as exc:
                logger.debug("Alias %s was not removed (%s)", alias, exc)
        for collection in self._client.get_collections().collections:
            if collection.name.startswith(f"{alias}__"):
                self._drop_collection(collection.name)

    def _search(self, workspace: str, vector: list[float], n: int, allowed_clearances: Sequence[str]) -> list[Hit]:
        self._require(workspace)
        if self.count(workspace) == 0:
            return []  # an empty workspace holds a one-dimensional placeholder: nothing to compare with
        response = self._client.query_points(
            self._alias(workspace),
            query=[float(x) for x in vector],
            limit=n,
            with_payload=True,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="clearance", match=models.MatchAny(any=list(allowed_clearances)))]
            ),
        )
        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(Hit(
                id=str(payload.get("id", point.id)),
                document=str(payload.pop("document", "")),
                metadata=payload,
                similarity=float(point.score),
            ))
        return hits

    # --- helpers -----------------------------------------------------------------------------------------------
    def _set_alias(self, alias: str, physical: str, previous: str | None) -> None:
        """Point the alias at `physical`, moving it from `previous` in one atomic request."""
        operations: list[Any] = []
        if previous is not None:
            operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        operations.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=physical, alias_name=alias)))
        self._client.update_collection_aliases(change_aliases_operations=operations)

    def _drop_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception as exc:
            logger.debug("Collection %s was not deleted (%s)", name, exc)

    def drop_namespace(self) -> None:
        """Remove every alias and collection of this prefix. For tests and for retiring an install's namespace."""
        for alias in list(self._aliases()):
            try:
                self._client.update_collection_aliases(
                    change_aliases_operations=[models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))]
                )
            except Exception as exc:
                logger.debug("Alias %s was not removed (%s)", alias, exc)
        for collection in self._client.get_collections().collections:
            if collection.name.startswith(f"{self._prefix}ws_"):
                self._drop_collection(collection.name)
