"""Install the native accelerator (rag_accel) the cheapest way available, without ever failing `make setup`.

In order: (1) it is already installed; (2) a prebuilt wheel in wheelhouse/ that pip considers compatible (the
`wheels` workflow in .github/workflows builds one per platform: macOS arm64 and x86_64, Linux x86_64 and aarch64;
each is a single abi3 wheel for Python 3.11 and later, so no toolchain is needed); (3) build it from crates/rag_accel
when `cargo` is on the PATH; (4) neither: say so plainly and carry on with the pure-Python scanner, which gives
the same verdicts but slower. RAG_ACCEL=required turns that last case into an error at startup.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rag_lab import ROOT

WHEELHOUSE = ROOT / "wheelhouse"
CRATE = ROOT / "crates" / "rag_accel" / "Cargo.toml"


def installed() -> bool:
    return importlib.util.find_spec("rag_accel") is not None


def has_wheel(wheelhouse: Path = WHEELHOUSE) -> bool:
    return wheelhouse.is_dir() and any(wheelhouse.glob("rag_accel-*.whl"))


def from_wheelhouse(wheelhouse: Path = WHEELHOUSE) -> bool:
    """pip picks the wheel whose tags fit this interpreter and platform; no index is consulted."""
    if not has_wheel(wheelhouse):
        return False
    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), "--upgrade", "rag-accel"],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        print(f"No wheel in {wheelhouse} fits this machine ({_last_line(done.stderr)}).")
    return done.returncode == 0


def from_source(crate: Path = CRATE) -> bool:
    if shutil.which("cargo") is None or not crate.is_file():
        return False
    if importlib.util.find_spec("maturin") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin>=1.7.0,<2"], check=False)
    done = subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--release", "--locked", "-m", str(crate)],
        env={**os.environ, "VIRTUAL_ENV": sys.prefix},
    )
    return done.returncode == 0


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "pip gave no reason"


def install() -> str:
    """Returns how it ended: "present", "wheel", "built" or "fallback"."""
    if installed():
        return "present"
    if from_wheelhouse():
        return "wheel"
    if from_source():
        return "built"
    return "fallback"


def main() -> int:
    outcome = install()
    if outcome == "present":
        print("Native accelerator: already installed.")
    elif outcome == "wheel":
        print("Native accelerator: installed from a prebuilt wheel in wheelhouse/.")
    elif outcome == "built":
        print("Native accelerator: built from crates/rag_accel.")
    else:
        print(
            "Native accelerator: NOT installed. No compatible wheel in wheelhouse/ and no Rust toolchain (cargo) to build one.\n"
            "  The pure-Python scanner will be used: same verdicts, slower, and 'make status' will show it as degraded.\n"
            "  To fix: download the wheel for your platform from the `wheels` workflow run into wheelhouse/ and rerun\n"
            "  'make setup', or install Rust (https://rustup.rs) and run 'make rust-build'.\n"
            "  Set RAG_ACCEL=required to refuse to start without it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
