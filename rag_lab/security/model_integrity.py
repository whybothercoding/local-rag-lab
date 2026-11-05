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

from rag_lab.config import settings


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
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "repo": "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        "revision": "faf4aa4225822f3bc6376869cb1164e8e3feedd0",
        "files": {
            "config.json": "c8ec081fdad2df991bf5abbf18418fec7a5cdaa421f60ffb060a30040b8c376f",
            "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
            "tokenizer.json": "fa685fc160bbdbab64058d4fc91b60e62d207e8dc60b9af5c002c5ab946ded00",
            "tokenizer_config.json": "0666eebf692422757e1dddf3c9fb1ded73ba3dc726c5828671fc89e45bf3609f",
            "model_optimized.onnx": "634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99",
        },
    },
    "BAAI/bge-reranker-v2-m3-int8": {
        "repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "revision": "6f5ff65298512715a1e669753bc754d2bc8f367b",
        "files": {
            "config.json": "122e922dcfed6503c8721e6fe1daf090340c3d95ca7f3aa3a72730b321a51cfd",
            "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
            "tokenizer.json": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
            "tokenizer_config.json": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
            "onnx/model_int8.onnx": "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
        },
    },
    "intfloat/multilingual-e5-large": {
        "repo": "Qdrant/multilingual-e5-large-onnx",
        "revision": "ac6781cd1cf88b8306a536d7c9d18a5bd57cc14b",
        "files": {
            "config.json": "1de8c3be1f344c0eefa4962480a006f8639f416dfafaa95a770e3cf4bceae6a4",
            "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
            "tokenizer.json": "f59925fcb90c92b894cb93e51bb9b4a6105c5c249fe54ce1c704420ac39b81af",
            "tokenizer_config.json": "f90024142df07163e5e6c5b9a6ad7c8c68b22a9112af11e3db4559a9ff90f737",
            "model.onnx": "1c09780c907c8a91a77a6ab1fd231f79e090d2907ca431223703dfebeed3d36c",
            "model.onnx_data": "0cf1883fee81c63819a44e2ba0efa51d4043d9759685a4ebebbde97e0623d15c",
        },
    },
    "BAAI/bge-reranker-v2-m3": {
        "repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "revision": "6f5ff65298512715a1e669753bc754d2bc8f367b",
        "files": {
            "config.json": "122e922dcfed6503c8721e6fe1daf090340c3d95ca7f3aa3a72730b321a51cfd",
            "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
            "tokenizer.json": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
            "tokenizer_config.json": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
            "onnx/model.onnx": "faae32b124a9d54afb7e89b5e9896e03c18a9552d56d1d6b273a709a83012486",
            "onnx/model.onnx_data": "f009aa6c6cf21986fd7e0021fa66b20ccce27abc6900a57c7109c8496811bcbe",
        },
    },
    # Full-precision weights on purpose. The int8 file of this model family returned scores that depend on the CPU
    # (the same true sentence: 0.93 on Apple silicon, 0.12 on x86 Linux), so a verdict could differ per machine;
    # fp32 gave identical scores on both. tests/test_nli.py refuses a quantized NLI file.
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": {
        "repo": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
        "files": {
            "config.json": "bc7b85f164a17c1b007d87fb99d3676f4a4d2e6511d2288dd3f5334dc0f34e6b",
            "special_tokens_map.json": "311de3f4eed9d76a43bf0d71f10e62e086ca65ccce9f15d5da0d2098bf519ecc",
            "tokenizer.json": "e23095eb61ba944c7be3a5d3e8ec19e37ce7ced0daa03550bde03e83c21b3f8a",
            "tokenizer_config.json": "25e9f84c6b43da7bff43a0aee8eeee2c8a39d246823d8464b9275ef704eab42f",
            "onnx/model.onnx": "e9238a33e07411eae1d1cef1d5f22e5f556717ce586c827581bb59598a45fe37",
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


def verify_model(model_name: str | None = None, cache_dir: str | None = None) -> Path:
    """Return the verified snapshot directory, or raise ModelIntegrityError."""
    model_name = settings.fastembed_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
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
