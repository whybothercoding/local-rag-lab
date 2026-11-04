import hashlib
from pathlib import Path

import pytest

import rag_lab.security.model_integrity as model_integrity
from rag_lab.security.model_integrity import ModelIntegrityError, verify_model


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


def test_fetch_models_verify_only_and_unpinned_error(monkeypatch, capsys):
    import rag_lab.ops.fetch_models as fetch_models
    with pytest.raises(ModelIntegrityError, match="no pinned SHA-256"):
        fetch_models.fetch_model("unpinned/model")

    monkeypatch.setattr("sys.argv", ["fetch_models.py", "--verify-only"])
    assert fetch_models.main() == 0
    assert "verified offline" in capsys.readouterr().out

    def fail_verify(*_args, **_kwargs):
        raise ModelIntegrityError("tampered weights")

    monkeypatch.setattr(fetch_models, "verify_model", fail_verify)
    assert fetch_models.main() == 1
    assert "tampered weights" in capsys.readouterr().err


def test_export_graph_mermaid_and_png_fallback(tmp_path, monkeypatch, mocker):
    import rag_lab.ops.export_graph as export_graph
    out_dir = tmp_path / "docs"
    mmd_file = out_dir / "architecture.mmd"
    png_file = out_dir / "architecture.png"
    monkeypatch.setattr(export_graph, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(export_graph, "MERMAID_FILE", str(mmd_file))
    monkeypatch.setattr(export_graph, "PNG_FILE", str(png_file))

    # 1. Without mmdc installed, render_png_locally returns False and .mmd is still written
    monkeypatch.setattr("rag_lab.ops.export_graph.shutil.which", lambda _: None)
    export_graph.export_mermaid_diagram()
    assert mmd_file.is_file()
    assert "guardrail" in mmd_file.read_text(encoding="utf-8")

    # 2. With mmdc available, subprocess.run is invoked and render_png_locally returns True
    monkeypatch.setattr("rag_lab.ops.export_graph.shutil.which", lambda _: "/usr/local/bin/mmdc")
    run_mock = mocker.patch("rag_lab.ops.export_graph.subprocess.run")
    assert export_graph.render_png_locally(str(mmd_file), str(png_file)) is True
    run_mock.assert_called_once()

