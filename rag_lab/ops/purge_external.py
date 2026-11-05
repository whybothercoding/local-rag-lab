"""Empty the external stores this install is configured to use. Part of `make factory-reset`.

Under VECTOR_STORE=qdrant|pgvector the indexed documents live on a server, and under CACHE_BACKEND=postgres the cached
answers do; deleting the local folders does not touch them, so a "zero-trace" reset must. Scope: every workspace in this
install's namespace (the Qdrant prefix, the pgvector schema) and the whole shared cache table. If a store cannot be
reached the line says NOT PURGED, loudly, and the command exits non-zero: it never claims a wipe that did not happen.
"""
from __future__ import annotations

import sys

from rag_lab.config import settings


def purge() -> tuple[list[str], bool]:
    """(report lines, everything configured was purged)."""
    lines: list[str] = []
    complete = True
    if settings.vector_store != "chroma":
        from rag_lab.storage.vector_store import get_vector_store, vector_store_status

        healthy, status = vector_store_status()
        if not healthy:
            lines.append(f"NOT PURGED: {status}. The indexed documents are still on that server.")
            complete = False
        else:
            store = get_vector_store()
            names = sorted(store.workspaces())
            for name in names:
                store.delete(name)
            left = store.workspaces()
            if left:
                lines.append(f"NOT PURGED: {len(left)} workspace(s) remain in {store.describe()}.")
                complete = False
            else:
                lines.append(f"Purged {len(names)} workspace(s) from {store.describe()}.")
    if settings.cache_backend == "postgres":
        from rag_lab.retrieval.semantic_cache import cache_backend_status, cache_stats, clear_cache

        healthy, where = cache_backend_status()
        if not healthy:
            lines.append(f"NOT PURGED: {where}. The cached answers are still there.")
            complete = False
        else:
            clear_cache()
            remaining = cache_stats()["total_entries"]
            if remaining:
                lines.append(f"NOT PURGED: {remaining} cached answer(s) remain in the {where}.")
                complete = False
            else:
                lines.append(f"Emptied the {where}.")
    if not lines:
        lines.append("No external store is configured: nothing to purge beyond the local folders.")
    return lines, complete


def main() -> int:
    lines, complete = purge()
    for line in lines:
        print(line)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
