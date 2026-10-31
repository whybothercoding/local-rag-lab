import hashlib
from pathlib import Path

import pytest

import model_integrity
from model_integrity import ModelIntegrityError, verify_model


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_snapshot(cache_dir: Path, repo: str, revision: str, files: dict[str, bytes]) -> Path:
    snap = cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    snap.mkdir(parents=True)
    for name, data in files.items():
        (snap / name).write_bytes(data)
    return snap


@pytest.fixture
def fake_pins(monkeypatch):
    files = {"model.onnx": b"weights-v1", "tokenizer.json": b"{}"}
    pins = {
        "test/model": {
            "repo": "test/fake-repo",
            "revision": "rev1",
            "files": {name: _sha(data) for name, data in files.items()},
        }
    }
    monkeypatch.setattr(model_integrity, "PINNED_MODELS", pins)
    return files


def test_verify_accepts_matching_files(tmp_path, fake_pins):
    _make_snapshot(tmp_path, "test/fake-repo", "rev1", fake_pins)
    assert verify_model("test/model", str(tmp_path)).name == "rev1"


def test_verify_rejects_tampered_file(tmp_path, fake_pins):
    _make_snapshot(tmp_path, "test/fake-repo", "rev1", {**fake_pins, "model.onnx": b"weights-EVIL"})
    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        verify_model("test/model", str(tmp_path))


def test_verify_rejects_missing_file(tmp_path, fake_pins):
    _make_snapshot(tmp_path, "test/fake-repo", "rev1", {"model.onnx": b"weights-v1"})
    with pytest.raises(ModelIntegrityError, match="Missing model file"):
        verify_model("test/model", str(tmp_path))


def test_verify_rejects_absent_cache(tmp_path, fake_pins):
    with pytest.raises(ModelIntegrityError, match="make models"):
        verify_model("test/model", str(tmp_path))


def test_verify_rejects_unpinned_model(tmp_path):
    with pytest.raises(ModelIntegrityError, match="no pinned"):
        verify_model("some/other-model", str(tmp_path))
