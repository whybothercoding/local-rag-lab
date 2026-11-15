# Day-to-day operations

## Health and tests

`make status` checks models, the accelerator, the LLM server, keys, workspaces, the vector store, the cache and audit health. It exits 1 if the vector store or the shared cache does not answer.
`make verify-models` re-checks the pinned model hashes.
`make test` runs the full suite of over 1,000 tests. Tests that need PostgreSQL, Qdrant or pgvector skip when the service is unreachable, and CI requires them.
`make test-offline` runs the suite with network sockets blocked.
`make lint` runs ruff and `make typecheck` runs mypy.
CI runs lint, mypy and the suite twice, online and offline, in about 20 minutes.

## Backups and retention

`make backup` creates a checksummed snapshot in `backups/`, `make backups` lists snapshots and `make backup-verify DIR=<dir>` verifies checksums, the audit chain and anchors.
`make retention` prunes expired audit records and idle checkpoints (default 183 days).
`make audit-export` exports the decrypted audit trail as JSONL. `make migrate-keys` moves key files into the OS keychain.

## Cleaning up

`make clean` removes caches and temporary artifacts.
`make factory-reset CONFIRM=yes` wipes every account, audit log, checkpoint, key, vector index and the source documents in `knowledge_base/` and `uploads/`, including any configured external stores.
Back up first, because a factory reset also deletes this workspace's files.

## Privacy

Queries are redacted for PII before they are logged or sent to a model. Nothing leaves the machine at runtime.
Audit records and idle checkpoints expire under `make retention`.

## Contributing rules

Every fix or feature needs tests and a green `make test`, `make test-offline`, `make lint` and `make typecheck`.
`AGENTS.md` must stay a byte-identical copy of `CLAUDE.md`, and a test fails otherwise.
