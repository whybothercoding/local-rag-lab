"""The native accelerator (rag_accel, Rust): preferred by default, with a fallback that is never quiet.

`RAG_ACCEL=preferred` (default) uses the native module when it is installed. When it is not, or a call into it
fails, the pure-Python scanner answers instead, with identical verdicts (tests/test_rust_accelerator.py), and the
process says so once on stderr; `make status` shows it as degraded. `RAG_ACCEL=required` turns that fallback into
an error: importing the sanitizer, which every entry point does, raises AcceleratorUnavailableError.
"""
from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from types import ModuleType

from rag_lab.config import settings

MODES = ("preferred", "required")

logger = logging.getLogger("rag_lab.accel")


class AcceleratorUnavailableError(RuntimeError):
    """RAG_ACCEL=required, and the native accelerator is missing or failed."""


@dataclass(frozen=True)
class AccelStatus:
    mode: str
    active: bool
    version: str = ""
    reason: str = ""
    degraded_calls: int = 0


_loaded: dict[str, object] = {}
_state = {"calls": 0, "noted": False}


def module() -> ModuleType | None:
    """The imported rag_accel module, or None where it is not installed (tried once per process)."""
    if "module" not in _loaded:
        try:
            _loaded["module"] = importlib.import_module("rag_accel")
            _loaded["error"] = ""
        except ImportError as exc:
            _loaded["module"] = None
            _loaded["error"] = str(exc) or "rag_accel is not installed"
    return _loaded["module"]  # type: ignore[return-value]


def _mode() -> str:
    mode = settings.rag_accel
    if mode not in MODES:
        raise ValueError(f"RAG_ACCEL must be one of {', '.join(MODES)}; got '{mode}'.")
    return mode


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("rag-accel")
    except Exception:
        return "unknown"


def status() -> AccelStatus:
    """What `make status` reports. Never raises."""
    mode = settings.rag_accel if settings.rag_accel in MODES else "preferred"
    if module() is None:
        return AccelStatus(mode, False, reason=str(_loaded.get("error", "")), degraded_calls=_state["calls"])
    degraded = _state["calls"]
    return AccelStatus(mode, degraded == 0, version=_version(),
                       reason="" if degraded == 0 else "a call into the native module failed and was answered in Python",
                       degraded_calls=degraded)


HOW_TO_FIX = (
    "Install a prebuilt wheel (put the CI artifact in wheelhouse/ and run 'make setup') or run 'make rust-build'."
)


def check() -> None:
    """Raise when RAG_ACCEL=required and the native accelerator is not usable. Called when the sanitizer is imported."""
    if _mode() == "required" and module() is None:
        raise AcceleratorUnavailableError(
            f"RAG_ACCEL=required, but the native accelerator is not available ({_loaded.get('error')}). {HOW_TO_FIX}"
        )


def note_degraded(reason: str) -> None:
    """Say, once per process, that the slower pure-Python scanner is answering."""
    if _state["noted"]:
        return
    _state["noted"] = True
    message = (f"the native accelerator is not in use ({reason}); the pure-Python scanner answers instead, with the "
               f"same verdicts but slower. {HOW_TO_FIX} RAG_ACCEL=required makes this an error.")
    logger.warning(message)
    print(f"Warning: {message}", file=sys.stderr)


def call_failed(exc: Exception) -> None:
    """A call into the native module raised. Required: that is an error. Preferred: answer in Python, and say so."""
    if _mode() == "required":
        raise AcceleratorUnavailableError(f"RAG_ACCEL=required, and a call into the native accelerator failed: {exc}") from exc
    _state["calls"] += 1
    note_degraded(f"a call into the native module failed: {exc}")
