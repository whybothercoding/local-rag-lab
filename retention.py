#!/usr/bin/env python3
"""Apply the retention policy to the audit log and idle checkpoint threads.

Default period is RETENTION_DAYS (183 days, about six months). Each run is recorded in the audit chain.
Run it on a schedule, for example from cron: make retention
"""
import argparse
import sys

from audit import prune
from checkpointer import prune_checkpoints
from config import RETENTION_DAYS
from semantic_cache import prune_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete records and threads older than the retention period.")
    parser.add_argument("--days", type=int, default=RETENTION_DAYS, help=f"Retention period (default {RETENTION_DAYS})")
    args = parser.parse_args(argv)
    if args.days < 1:
        print("Retention must be at least one day.", file=sys.stderr)
        return 1

    audit_result = prune(args.days)
    print(
        f"Audit log: removed {audit_result['removed']} record(s) older than {audit_result['cutoff']}. "
        f"Chain base is now seq {audit_result['base_seq']}."
    )
    threads = prune_checkpoints(args.days)
    print(f"Checkpoints: removed {threads} idle thread(s). Threads awaiting review were kept.")
    cached = prune_cache(args.days)
    print(f"Cache: removed {cached} expired entry(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
