#!/usr/bin/env python3
"""Download the pinned embedding model into ./models/ and verify it by SHA-256.

Run once on a connected machine with `make models`. Afterwards the application
runs offline: config.py sets HF_HUB_OFFLINE, so nothing is fetched at runtime.
Use `--verify-only` to check an existing cache without any network access.
"""
import argparse
import os
from pathlib import Path
import sys

# This script is the only place network access is re-enabled. It must be set before
# huggingface_hub is imported, because the library reads the flag at import time.
os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import snapshot_download

from config import FASTEMBED_MODEL, MODELS_DIR, RERANKER_MODEL
from model_integrity import PINNED_MODELS, ModelIntegrityError, verify_model


def fetch_model(model_name: str = FASTEMBED_MODEL, cache_dir: str = MODELS_DIR):
    spec = PINNED_MODELS.get(model_name)
    if spec is None:
        raise ModelIntegrityError(f"Model '{model_name}' has no pinned SHA-256 entries; refusing to download it.")
    snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        cache_dir=cache_dir,
        allow_patterns=list(spec["files"]),
    )
    ref_dir = Path(cache_dir) / f"models--{spec['repo'].replace('/', '--')}" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "main").write_text(spec["revision"], encoding="utf-8")
    return verify_model(model_name, cache_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify pinned model weights.")
    parser.add_argument("--verify-only", action="store_true", help="Verify the cache without network access")
    args = parser.parse_args()

    try:
        models_to_check = [FASTEMBED_MODEL, RERANKER_MODEL]
        if args.verify_only:
            for m in models_to_check:
                path = verify_model(m, MODELS_DIR)
                print(f"Model '{m}' verified offline at {path}")
        else:
            for m in models_to_check:
                path = fetch_model(m, MODELS_DIR)
                print(f"Model '{m}' downloaded and verified at {path}")
    except ModelIntegrityError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
