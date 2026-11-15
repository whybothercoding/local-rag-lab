# Security model

The system fails closed. A request without a valid identity, permission or clearance is refused, not downgraded.

## Access control

Policy v3 RBAC defines six roles: owner, admin, reviewer, trusted, member and none.
The roles bundle three permissions: `query`, `review` and `auto_approve`. Owners and admins can also administer accounts.
Permissions resolve in order: the account's workspace grant, its all-workspaces grant, its groups' grants (any `none` wins), its own role, then the default role.
On a fresh install the default role is `trusted`.
Only authenticated administrators change accounts, roles, groups and grants. Only an owner assigns the owner role, and the last owner cannot be removed.
Document clearance is separate from roles. The clearance levels are public, internal, confidential and secret.

## Clearance enforcement

Documents with an unrecognized clearance tag are rejected at ingestion.
Dense, sparse and graph retrieval filter by the allowed clearances, and the base `VectorStore.query()` then runs `guard_hits()`, which drops any hit without a valid clearance. A new storage adapter cannot skip this guard.

## Sessions, keys and second factor

Sessions use signed, revocable tokens.
Keys live in the OS keychain, or in owner-only files when no keychain exists (`KEY_STORE=auto|file|keyring`).
Optional TOTP is available for owners and admins (`make totp-enroll`). A code can be spent only once, for a newer time step.
Owners reset other accounts. A sole owner who is locked out recovers with `make recover`, a break-glass command run at this machine's terminal and audited. `LOCAL_RECOVERY=off` disables it.
State, cache entries and checkpoints are sealed with AES-256-GCM.

## Audit log

Every query, guardrail check, retrieval and human decision is written to an append-only SQLite audit log.
Records are chained with HMAC-SHA256, and database triggers block updates and deletes.
`make audit-verify` checks the chain. `make audit-anchor` records the head hash, and `AUDIT_ANCHOR_COPY_DIR` writes a second copy on another disk so truncation is detected.

## Prompt injection and PII

Questions are normalized (Unicode NFKC, zero-width characters stripped) and matched against the patterns in `rag_lab/security/injection_patterns.tsv`. A match ends the run before retrieval.
Retrieved chunks are screened again before generation. Poisoned chunks are dropped and replaced by the next clean candidate.
PII such as card numbers, IBANs and phone numbers is masked before it reaches logs, prompts or checkpoints.
After editing the patterns file, run `make rust-build`.

## Known limits

Chunk text in Chroma, FTS5, Qdrant and pgvector is stored unencrypted. Access control is enforced by the application, so use disk encryption such as FileVault or LUKS on shared hosts.
The full threat model is in `docs/threat-model.md`.
