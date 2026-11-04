"""`make setup` installs the native accelerator the cheapest way available and never fails for lack of it."""
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from rag_lab.ops import install_accel

ROOT = Path(__file__).resolve().parent.parent

# The file names maturin gives the four wheels the `wheels` workflow builds (checked against real builds on
# macOS arm64 and, in a manylinux container, Linux aarch64).
WHEELS = {
    "macos-arm64": "rag_accel-0.1.0-cp311-abi3-macosx_11_0_arm64.whl",
    "macos-x86_64": "rag_accel-0.1.0-cp311-abi3-macosx_10_12_x86_64.whl",
    "linux-x86_64": "rag_accel-0.1.0-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "linux-aarch64": "rag_accel-0.1.0-cp311-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
}
MACHINES = {
    "macos-arm64": ["macosx_11_0_arm64", "macosx_14_0_arm64"],
    "macos-x86_64": ["macosx_10_12_x86_64", "macosx_10_15_x86_64"],
    "linux-x86_64": ["manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64"],
    "linux-aarch64": ["manylinux2014_aarch64", "manylinux_2_17_aarch64", "manylinux_2_28_aarch64"],
}


def fake_wheel(directory: Path, filename: str) -> None:
    """An installable wheel with nothing in it: enough for pip to choose between candidates."""
    with zipfile.ZipFile(directory / filename, "w") as wheel:
        info = "rag_accel-0.1.0.dist-info"
        wheel.writestr(f"{info}/METADATA", "Metadata-Version: 2.1\nName: rag-accel\nVersion: 0.1.0\n")
        wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: cp311-abi3-any\n")
        wheel.writestr(f"{info}/RECORD", "")


def chosen_by_pip(wheelhouse: Path, platforms: list[str], python: str, tmp_path: Path) -> str | None:
    command = [
        sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--no-deps", "--no-index",
        "--only-binary=:all:", "--find-links", str(wheelhouse), "--target", str(tmp_path / "t"),
        "--implementation", "cp", "--python-version", python, "--abi", "abi3", "--abi", f"cp{python.replace('.', '')}",
        "--report", "-", "--quiet", "rag-accel",
    ]
    for platform in platforms:
        command += ["--platform", platform]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    report = json.loads(done.stdout)
    return report["install"][0]["download_info"]["url"].rsplit("/", 1)[-1]


@pytest.fixture(scope="module")
def wheelhouse(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wheelhouse")
    for filename in WHEELS.values():
        fake_wheel(directory, filename)
    return directory


@pytest.mark.parametrize("python", ["3.11", "3.14"])
@pytest.mark.parametrize("machine", list(MACHINES))
def test_pip_picks_the_wheel_for_each_platform_and_any_python_from_3_11(wheelhouse, machine, python, tmp_path):
    """One abi3 wheel per platform serves 3.11 and every later version; no wheel is picked for the wrong machine."""
    assert chosen_by_pip(wheelhouse, MACHINES[machine], python, tmp_path) == WHEELS[machine]


def test_a_platform_without_a_wheel_gets_none(wheelhouse, tmp_path):
    assert chosen_by_pip(wheelhouse, ["win_amd64"], "3.12", tmp_path) is None


def test_python_before_3_11_gets_none(wheelhouse, tmp_path):
    assert chosen_by_pip(wheelhouse, MACHINES["macos-arm64"], "3.10", tmp_path) is None


# --- the order of the attempts ---------------------------------------------------------------------------

@pytest.fixture
def steps(monkeypatch):
    calls = []
    results = {"installed": False, "wheel": False, "source": False}

    monkeypatch.setattr(install_accel, "installed", lambda: calls.append("installed") or results["installed"])
    monkeypatch.setattr(install_accel, "from_wheelhouse", lambda *a: calls.append("wheel") or results["wheel"])
    monkeypatch.setattr(install_accel, "from_source", lambda *a: calls.append("source") or results["source"])
    return calls, results


def test_already_installed_does_nothing(steps):
    calls, results = steps
    results["installed"] = True
    assert install_accel.install() == "present" and calls == ["installed"]


def test_a_wheel_is_tried_before_building(steps):
    calls, results = steps
    results["wheel"] = True
    assert install_accel.install() == "wheel" and calls == ["installed", "wheel"]


def test_cargo_is_used_only_when_no_wheel_fits(steps):
    calls, results = steps
    results["source"] = True
    assert install_accel.install() == "built" and calls == ["installed", "wheel", "source"]


def test_with_neither_it_falls_back_loudly_and_does_not_fail_setup(steps, capsys):
    assert install_accel.install() == "fallback"
    assert install_accel.main() == 0, "make setup must not fail for lack of a Rust toolchain"
    out = capsys.readouterr().out
    assert "NOT installed" in out and "wheelhouse/" in out and "rustup" in out and "degraded" in out
    assert "RAG_ACCEL=required" in out


def test_wheels_are_found_by_name(tmp_path):
    assert not install_accel.has_wheel(tmp_path / "missing")
    assert not install_accel.has_wheel(tmp_path)
    (tmp_path / "other-1.0-py3-none-any.whl").write_bytes(b"")
    assert not install_accel.has_wheel(tmp_path)
    fake_wheel(tmp_path, WHEELS["macos-arm64"])
    assert install_accel.has_wheel(tmp_path)


def test_without_cargo_there_is_no_build(monkeypatch):
    monkeypatch.setattr(install_accel.shutil, "which", lambda name: None)
    assert install_accel.from_source() is False


def test_an_empty_wheelhouse_is_not_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(install_accel.subprocess, "run", lambda *a, **k: pytest.fail("pip must not run without a wheel"))
    assert install_accel.from_wheelhouse(tmp_path) is False


# --- the build files ---------------------------------------------------------------------------------------

def test_the_crate_builds_one_abi3_wheel_from_locked_dependencies():
    cargo = tomllib.loads((ROOT / "crates/rag_accel/Cargo.toml").read_text())
    assert "abi3-py311" in cargo["dependencies"]["pyo3"]["features"]
    assert "extension-module" not in cargo["dependencies"]["pyo3"]["features"]
    project = tomllib.loads((ROOT / "crates/rag_accel/pyproject.toml").read_text())
    assert project["build-system"]["build-backend"] == "maturin" and project["project"]["requires-python"] == ">=3.11"
    assert not (ROOT / "crates/rag_accel/src/chunker.rs").exists(), "the native chunker was removed (0.05% of ingest)"
    makefile = (ROOT / "Makefile").read_text()
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in makefile and "--locked" in makefile
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in (ROOT / ".github/workflows/ci.yml").read_text()


def test_the_wheels_workflow_covers_the_four_platforms_and_smoke_tests_each():
    workflow = yaml.safe_load((ROOT / ".github/workflows/wheels.yml").read_text())
    names = {entry["name"] for entry in workflow["jobs"]["build"]["strategy"]["matrix"]["include"]}
    assert names == set(WHEELS)
    steps = workflow["jobs"]["build"]["steps"]
    build = next(s for s in steps if "maturin-action" in s.get("uses", ""))
    assert "--locked" in build["with"]["args"] and "--find-interpreter" not in build["with"]["args"]
    assert any("Smoke test" in s.get("name", "") for s in steps)
    assert any("upload-artifact" in s.get("uses", "") for s in steps)


def test_ci_requires_the_accelerator():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("RAG_ACCEL: required") == 2, "both test runs refuse to fall back"
