"""PostgreSQL with the pgvector extension as the vector store (VECTOR_STORE=pgvector). Needs no new Python package.

Layout, in one schema (PGVECTOR_SCHEMA, `public` by default):

    rag_workspaces(workspace PRIMARY KEY, fingerprint, created_at)    -- one row per workspace, even an empty one
    rag_chunks(workspace, id, document, metadata jsonb, clearance, embedding vector)   -- PRIMARY KEY (workspace, id)

All workspaces share the one table, which is why every query filters on `workspace` and why VectorStore.query() checks
the workspace of each hit again. A publish is one transaction (delete the workspace's rows, insert the new ones, update
its fingerprint): readers see the old index until the commit, and a failure rolls back to it.

Scores. `<=>` is pgvector's cosine *distance*, so `1 - (embedding <=> query)` is the cosine similarity, the same scale as
Chroma's `1 - distance`.

No ANN index is created. The column has no fixed dimension (the model can change), HNSW needs one, and an exact scan of
a workspace's rows is what Chroma's results are compared with. For workspaces beyond a few hundred thousand chunks add an
HNSW index on the cast column yourself (docs/scaling.md). There is no unfiltered fallback: a failing filter makes dense
retrieval unavailable (labelled) instead of asking for everything.
"""
from __future__ import annotations

import atexit
import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from rag_lab.config import settings
from rag_lab.retrieval.embedding_provider import FINGERPRINT_KEY
from rag_lab.storage.vector_store import Hit, VectorStore, VectorStoreError, WorkspaceNotIndexed

_SCHEMA = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_BATCH = 500


def _literal(vector: Sequence[float]) -> str:
    """pgvector's text form of a vector: '[0.1,0.2,...]'. Passed as a parameter and cast with ::vector."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgvectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, uri: str, schema: str = "public") -> None:
        if not uri:
            raise VectorStoreError(
                "VECTOR_STORE=pgvector needs PGVECTOR_URI, or POSTGRES_PASSWORD in .env for the compose profile "
                "(`make scale-up PROFILE=pgvector`)."
            )
        if not _SCHEMA.fullmatch(schema):
            raise VectorStoreError(f"PGVECTOR_SCHEMA must be a plain lowercase identifier; got '{schema}'.")
        self._schema = schema
        self._where = uri
        self._pool = ConnectionPool(conninfo=uri, min_size=1, max_size=max(2, settings.checkpoint_pool_size), open=True,
                                    timeout=15, kwargs={"autocommit": False})
        # The pool's worker threads must be stopped before the interpreter finalises (Python 3.14 raises otherwise).
        atexit.register(self.close)
        try:
            self._pool.wait(timeout=15)
            self._create_tables()
        except Exception as exc:
            self._pool.close()
            raise VectorStoreError(
                f"Could not prepare pgvector in {uri.split('@')[-1]} ({type(exc).__name__}: {exc}). The server needs the "
                "pgvector extension: use the `pgvector` compose profile (image pgvector/pgvector), not plain postgres."
            ) from exc

    def describe(self) -> str:
        from rag_lab.storage.vector_store import safe_location

        return f"pgvector at {safe_location(self._where)} (schema {self._schema})"

    def close(self) -> None:
        self._pool.close()

    # --- SQL helpers ----------------------------------------------------------------------------------------------
    def _t(self, table: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(self._schema), sql.Identifier(table))

    def _create_tables(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self._schema)))
            conn.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (workspace text PRIMARY KEY, fingerprint text, "
                "created_at timestamptz NOT NULL DEFAULT now())").format(self._t("rag_workspaces")))
            conn.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (workspace text NOT NULL, id text NOT NULL, document text NOT NULL, "
                "metadata jsonb NOT NULL, clearance text NOT NULL, embedding vector NOT NULL, "
                "PRIMARY KEY (workspace, id))").format(self._t("rag_chunks")))
            conn.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (workspace, clearance)").format(
                sql.Identifier(f"rag_chunks_clearance_{self._schema}"[:63]), self._t("rag_chunks")))

    def _scalar(self, query: sql.Composed, params: tuple) -> Any:
        with self._pool.connection() as conn:
            row = conn.execute(query, params).fetchone()
        return row[0] if row else None

    # --- the interface ------------------------------------------------------------------------------------------------
    def workspaces(self) -> set[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(sql.SQL("SELECT workspace FROM {}").format(self._t("rag_workspaces"))).fetchall()
        return {r[0] for r in rows}

    def exists(self, workspace: str) -> bool:
        return bool(self._scalar(sql.SQL("SELECT 1 FROM {} WHERE workspace = %s").format(self._t("rag_workspaces")), (workspace,)) == 1)

    def _require(self, workspace: str) -> None:
        if not self.exists(workspace):
            raise WorkspaceNotIndexed(workspace)

    def ensure(self, workspace: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(sql.SQL("INSERT INTO {} (workspace) VALUES (%s) ON CONFLICT DO NOTHING").format(
                self._t("rag_workspaces")), (workspace,))

    def count(self, workspace: str) -> int:
        self._require(workspace)
        return int(self._scalar(sql.SQL("SELECT count(*) FROM {} WHERE workspace = %s").format(self._t("rag_chunks")), (workspace,)))

    def collection_metadata(self, workspace: str) -> dict[str, Any] | None:
        self._require(workspace)
        fingerprint = self._scalar(sql.SQL("SELECT fingerprint FROM {} WHERE workspace = %s").format(self._t("rag_workspaces")), (workspace,))
        return {FINGERPRINT_KEY: fingerprint} if fingerprint else {}

    def read_vectors(self, workspace: str) -> Iterator[tuple[str, list[float]]]:
        self._require(workspace)
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL("SELECT document, embedding::real[] FROM {} WHERE workspace = %s ORDER BY id").format(self._t("rag_chunks")),
                (workspace,),
            ).fetchall()
        for document, vector in rows:
            yield str(document), [float(x) for x in vector]

    def publish(self, workspace, ids, documents, metadatas, vectors, fingerprint) -> None:
        # Checked before the transaction opens: pgvector would store vectors of mixed sizes in an unsized column.
        if not (len(ids) == len(documents) == len(metadatas) == len(vectors)):
            raise ValueError("ids, documents, metadatas and vectors must be the same length")
        if not ids:
            raise ValueError("nothing to publish")
        sizes = {len(v) for v in vectors}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(f"vectors must all have the same non-zero size; got {sorted(sizes)}")
        rows = [
            (workspace, cid, doc, Jsonb({**meta, "id": cid, "workspace": workspace}), str(meta.get("clearance", "")).strip().lower(), _literal(vec))
            for cid, doc, meta, vec in zip(ids, documents, metadatas, vectors)
        ]
        with self._pool.connection() as conn:  # one transaction: commit on success, roll back on any error
            conn.execute(
                sql.SQL("INSERT INTO {} (workspace, fingerprint) VALUES (%s, %s) "
                        "ON CONFLICT (workspace) DO UPDATE SET fingerprint = EXCLUDED.fingerprint").format(self._t("rag_workspaces")),
                (workspace, fingerprint),
            )
            conn.execute(sql.SQL("DELETE FROM {} WHERE workspace = %s").format(self._t("rag_chunks")), (workspace,))
            insert = sql.SQL("INSERT INTO {} (workspace, id, document, metadata, clearance, embedding) "
                             "VALUES (%s, %s, %s, %s, %s, %s::vector)").format(self._t("rag_chunks"))
            with conn.cursor() as cursor:
                for start in range(0, len(rows), _BATCH):
                    cursor.executemany(insert, rows[start:start + _BATCH])

    def delete(self, workspace: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(sql.SQL("DELETE FROM {} WHERE workspace = %s").format(self._t("rag_chunks")), (workspace,))
            conn.execute(sql.SQL("DELETE FROM {} WHERE workspace = %s").format(self._t("rag_workspaces")), (workspace,))

    def _search(self, workspace: str, vector: list[float], n: int, allowed_clearances: Sequence[str]) -> list[Hit]:
        self._require(workspace)
        literal = _literal(vector)
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL("SELECT id, document, metadata, clearance, 1 - (embedding <=> %s::vector) AS similarity FROM {} "
                        "WHERE workspace = %s AND clearance = ANY(%s) ORDER BY embedding <=> %s::vector LIMIT %s").format(self._t("rag_chunks")),
                (literal, workspace, [c.strip().lower() for c in allowed_clearances], literal, n),
            ).fetchall()
        hits = []
        for cid, document, metadata, clearance, similarity in rows:
            meta = dict(metadata if isinstance(metadata, dict) else json.loads(metadata))
            meta["clearance"], meta["workspace"] = clearance, workspace  # the filtered columns are the truth
            hits.append(Hit(id=str(cid), document=str(document), metadata=meta, similarity=float(similarity)))
        return hits

    def drop_namespace(self) -> None:
        """Remove this schema and everything in it. For tests and for retiring an install's namespace.

        Idempotent: a fixture teardown may call it again after the test dropped the namespace itself.
        """
        if self._pool.closed:
            return
        with self._pool.connection() as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self._schema)))
        self._pool.close()
        # The closed store must not stay cached: every later get_vector_store() would hand out a dead pool.
        from rag_lab.storage import vector_store

        for key, store in list(vector_store._INSTANCES.items()):
            if key[0] == "pgvector" and store is self:
                del vector_store._INSTANCES[key]
