"""SHA-256 verification for locally cached model weights.

Every model the application loads must be pinned here. Files are checked before
they are handed to ONNX Runtime, and unpinned or altered weights are refused.

The pins were taken from a download made on this project's first setup. They
protect against tampering after that download; they do not vouch for the
download itself. Review any change to a pin deliberately.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from config import FASTEMBED_MODEL, MODELS_DIR, RERANKER_MODEL


class ModelIntegrityError(RuntimeError):
    """A model is missing, unpinned, or does not match its pinned hashes."""


# Model name -> Hugging Face repo, exact revision (commit), and SHA-256 of each file the loader needs.
PINNED_MODELS: dict[str, dict] = {
    "BAAI/bge-small-en-v1.5": {
        "repo": "Qdrant/bge-small-en-v1.5-onnx-Q",
        "revision": "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        "files": {
            "config.json": "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
            "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
            "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
        },
    },
    "Xenova/ms-marco-MiniLM-L-6-v2": {
        "repo": "Xenova/ms-marco-MiniLM-L-6-v2",
        "revision": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        "files": {
            "config.json": "d827779a72d27ae68cf878a6fc2e954542663fe21ca515d9f4783fc96be2d37e",
            "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
            "onnx/model.onnx": "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_path(repo: str, revision: str, cache_dir: str) -> Path:
    """Path of one revision inside the Hugging Face cache layout used by fastembed."""
    return Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def verify_model(model_name: str = FASTEMBED_MODEL, cache_dir: str = MODELS_DIR) -> Path:
    """Return the verified snapshot directory, or raise ModelIntegrityError."""
    spec = PINNED_MODELS.get(model_name)
    if spec is None:
        raise ModelIntegrityError(f"Model '{model_name}' has no pinned SHA-256 entries; refusing to load it.")

    snapshot = snapshot_path(spec["repo"], spec["revision"], cache_dir)
    if not snapshot.is_dir():
        raise ModelIntegrityError(
            f"Model '{model_name}' is not cached in {cache_dir}. Run 'make models' once on a connected machine."
        )

    for filename, expected in spec["files"].items():
        path = snapshot / filename
        if not path.is_file():
            raise ModelIntegrityError(f"Missing model file: {path}. Run 'make models' again.")
        actual = sha256_file(path)
        if actual != expected:
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}. "
                f"Delete {cache_dir} and run 'make models' again."
            )
    return snapshot
