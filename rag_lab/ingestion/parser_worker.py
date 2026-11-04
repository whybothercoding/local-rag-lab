#!/usr/bin/env python3
"""Child process that extracts text from one document. Started by parsers.extract_text_from_file.

Prints {"text": ...} as the last stdout line. On failure, prints the reason to stderr and exits with 2.
"""
import json
import sys
from pathlib import Path

from rag_lab.ingestion.parsers import _extract_in_process, _limit_child_resources


def main() -> int:
    _limit_child_resources()
    try:
        text = _extract_in_process(Path(sys.argv[1]))
    except Exception as exc:
        print(str(exc) or type(exc).__name__, file=sys.stderr)
        return 2
    print(json.dumps({"text": text}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
