"""Context sanitizer protecting against indirect prompt injection in retrieved documents.

Scans retrieved candidate chunks for adversarial instruction overrides, role-spoofing
delimiters, jailbreak directives, and exfiltration payloads before they enter the LLM prompt.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

# The one pattern list both scanners use: this module reads it; crates/rag_accel compiles it in.
PATTERNS_PATH = Path(__file__).resolve().parent / "injection_patterns.tsv"


def _load_patterns(path: Path = PATTERNS_PATH) -> tuple[list[tuple[re.Pattern[str], str]], frozenset[str]]:
    """Compile scope<TAB>reason<TAB>regex lines. Returns (patterns in file order, document-only reasons)."""
    patterns: list[tuple[re.Pattern[str], str]] = []
    document_only: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        scope, reason, regex = line.split("\t")
        if scope not in ("any", "document"):
            raise ValueError(f"unknown pattern scope {scope!r} for {reason}")
        patterns.append((re.compile(regex, re.IGNORECASE), reason))
        if scope == "document":
            document_only.add(reason)
    return patterns, frozenset(document_only)


_INJECTION_PATTERNS, _DOCUMENT_ONLY_REASONS = _load_patterns()


# Try importing native Rust accelerator, fallback to pure Python
try:
    import rag_accel
    RUST_ACCEL = True
except ImportError:
    RUST_ACCEL = False


_INVISIBLE = dict.fromkeys(
    [0x00AD, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0xFEFF]
    + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A)),
)

# Reasons scoped 'document' in the pattern file only make sense for retrieved documents: a person
# asking about web markup in a question is not attacking the model.


def normalize_for_scan(text: str) -> str:
    """Fold look-alike characters, drop invisible ones, and collapse whitespace before pattern matching.

    Without this, "ignore  previous\u200b instructions", full-width letters, or accented spellings slip
    past the patterns.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    # Strip accents so "PRÉCÉDENTES" and "Αγνόησε" meet the accent-free patterns. Case is kept: the
    # patterns are case-insensitive, and the matched text is reported back as written.
    decomposed = unicodedata.normalize("NFKD", folded)
    stripped = unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))
    return re.sub(r"\s+", " ", stripped)


def scan_user_input(text: str) -> tuple[bool, str | None]:
    """Scan a user's question for instruction-override and role-spoofing attempts.

    Uses the same pattern set as retrieved documents, minus document-only markup rules, after
    Unicode and whitespace normalization. Returns (is_safe, violation_reason).
    """
    normalized = normalize_for_scan(text)
    for pattern, reason in _INJECTION_PATTERNS:
        if reason in _DOCUMENT_ONLY_REASONS:
            continue
        match = pattern.search(normalized)
        if match:
            return False, f"{reason}:{match.group(0)}"
    return True, None


def scan_chunk(text: str) -> tuple[bool, str | None]:
    """Scan a text chunk for indirect prompt injection payloads.

    Uses native Rust Aho-Corasick & regex accelerator when available,
    falling back to pure Python reference implementation.

    Returns:
        (is_safe, violation_reason)
    """
    text = normalize_for_scan(text)
    if RUST_ACCEL:
        try:
            verdict: tuple[bool, str | None] = rag_accel.scan_chunk(text)
            return verdict
        except Exception:
            pass

    for pattern, reason in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return False, f"{reason}:{match.group(0)}"
    return True, None


def sanitize_chunks(
    documents: list[str],
    citations: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Inspect and filter document chunks and citations.

    Maintains 1-to-1 correspondence between documents and citations. Any chunk
    exhibiting prompt injection signatures is purged from the context.

    Returns:
        (clean_documents, clean_citations, dropped_violations)
    """
    clean_docs: list[str] = []
    clean_cites: list[str] = []
    dropped: list[dict[str, Any]] = []

    # Handle cases where lengths might differ defensively
    max_len = max(len(documents), len(citations))
    for i in range(max_len):
        doc = documents[i] if i < len(documents) else ""
        cite = citations[i] if i < len(citations) else f"[CHUNK-{i}] unknown"

        is_safe, reason = scan_chunk(doc)
        if is_safe:
            clean_docs.append(doc)
            clean_cites.append(cite)
        else:
            dropped.append({
                "citation": cite,
                "reason": reason or "unknown_violation",
                "preview": doc[:120],
            })

    return clean_docs, clean_cites, dropped
