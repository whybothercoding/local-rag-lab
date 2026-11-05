"""Create .env with a random PostgreSQL password the first time `make up` runs.

An existing .env is never opened, read or changed: this module only creates the file when it is absent.
"""
from __future__ import annotations

import os
import secrets
import sys

from rag_lab import ROOT

TEMPLATE = """# Created by 'make up'. This file is git-ignored; keep it that way.
POSTGRES_USER=rag_user
POSTGRES_DB=rag_checkpoints
POSTGRES_PASSWORD={password}

# To use this PostgreSQL server as the checkpoint store, uncomment (the app reads this file too):
# DB_URI=postgresql://${{POSTGRES_USER}}:${{POSTGRES_PASSWORD}}@localhost:5432/${{POSTGRES_DB}}
"""


def main() -> int:
    target = ROOT / ".env"
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return 0  # not read, not touched
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(TEMPLATE.format(password=secrets.token_urlsafe(24)))
    print(f"Created {target.name} with a random PostgreSQL password (mode 600).")
    print("If a database volume from an earlier version exists, run 'make db-password' once so it accepts this password.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
