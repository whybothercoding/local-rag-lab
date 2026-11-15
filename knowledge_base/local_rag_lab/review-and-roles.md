# Human review and separation of duties

An answer is released automatically only if the user holds the `auto_approve` permission and the answer is grounded, relevant and free of blocking degradation.
An abstention follows the same rule: a clean one releases, and one that follows a broken search waits for review.
Otherwise the pipeline pauses with `interrupt()` and waits for a reviewer.

## Who can review

A reviewer needs the `review` permission, must not be the requester, and must hold a clearance that covers the thread's clearance.
This separation of duties is enforced inside the pipeline node, not only in the UI.
The UI disables input for a conversation while its review is pending.

## Commands

`make reviews` lists threads awaiting approval.
`make review` walks through a paused thread interactively.
`make approve THREAD=<id> ACCOUNT=<reviewer> REASON="..."` approves, and `make reject` rejects.
`make audit-feedback` shows answers rejected by reviewers with their reasons.
`make audit-harvest` exports rejected answers as candidate cases for `eval/golden.json`, which closes a feedback loop between review and evaluation.

## Managing accounts

`make add-user ACCOUNT=<name>` creates an account. The first account must be an administrator and becomes the owner.
`make set-role`, `make grant`, `make revoke`, `make create-group`, `make grant-group` and `make set-clearance` manage access.
Once the first administrator exists, user and workspace commands need `ADMIN=<admin>`.
`make team-mode ADMIN=<admin>` changes the default role from trusted to member, so answers need a reviewer.
