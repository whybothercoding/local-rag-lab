"""Files the package needs must be tracked: a .gitignore pattern once swallowed rag_lab/lang/data/ (never committed)."""
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def _ignored(paths: list[Path]) -> list[str]:
    relative = [str(p.relative_to(ROOT)) for p in paths]
    done = subprocess.run(["git", "check-ignore", "--no-index", *relative], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.split()


def test_package_data_is_not_gitignored():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    files: list[Path] = []
    for package, patterns in config["tool"]["setuptools"]["package-data"].items():
        directory = ROOT / package.replace(".", "/")
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            assert matches, f"package-data {package}: {pattern} matches nothing"
            files += matches
    assert any("lang/data" in str(f) for f in files), "the language data files are package data"
    assert _ignored(files) == []


def test_runtime_data_is_still_ignored_at_the_root_only():
    assert _ignored([ROOT / "data" / "audit.db"]) == ["data/audit.db"]
    assert _ignored([ROOT / "rag_lab" / "lang" / "data" / "en.json"]) == []


def test_the_rust_lockfile_is_tracked_for_reproducible_wheels():
    assert _ignored([ROOT / "crates" / "rag_accel" / "Cargo.lock"]) == []


def test_the_license_is_declared_once_and_agrees_everywhere():
    """The README said Apache-2.0 while pyproject.toml said MIT and no LICENSE file existed."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["license"]["text"] == "Apache-2.0"
    assert "Apache-2.0" in (ROOT / "README.md").read_text(encoding="utf-8").rsplit("## License", 1)[1]
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text.splitlines()[0] and "Version 2.0, January 2004" in text
