"""Context sanitizer protecting against indirect prompt injection in retrieved documents.

Scans retrieved candidate chunks for adversarial instruction overrides, role-spoofing
delimiters, jailbreak directives, and exfiltration payloads before they enter the LLM prompt.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Compiled heuristic patterns for indirect prompt injection
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 1. Direct instruction overrides & directive wipes
    (
        re.compile(
            r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|rules|guidelines|prompts)\b",
            re.IGNORECASE,
        ),
        "instruction_override",
    ),
    (
        re.compile(
            r"\bdisregard\s+(all\s+)?(prior|previous|above)\s+(directives|instructions|rules|prompts)\b",
            re.IGNORECASE,
        ),
        "directive_wipe",
    ),
    (
        re.compile(r"\bsystem\s+prompt\s+override\b", re.IGNORECASE),
        "system_override",
    ),
    (
        re.compile(r"\bbypass\s+(all\s+)?security\s+(rules|filters|guidelines)\b", re.IGNORECASE),
        "security_bypass",
    ),
    (
        re.compile(r"\bnew\s+(system\s+)?instruction\s*:", re.IGNORECASE),
        "new_instruction",
    ),
    (
        re.compile(
            r"\bdo\s+not\s+follow\s+(the\s+)?(original|initial|prior)\s+instructions\b",
            re.IGNORECASE,
        ),
        "instruction_rejection",
    ),
    (
        re.compile(r"\byou\s+are\s+now\s+in\s+([a-z0-9_-]+\s+)?mode\b", re.IGNORECASE),
        "role_override",
    ),
    (
        re.compile(r"\b(jailbreak|unfiltered\s+mode|dan\s+mode)\b", re.IGNORECASE),
        "jailbreak_signature",
    ),
    # 2. Role spoofing and delimiter manipulation
    (
        re.compile(r"<\|im_(start|end)\|>", re.IGNORECASE),
        "chatml_delimiter_spoofing",
    ),
    (
        re.compile(r"\[(?:SYSTEM|ADMIN|DEVELOPER)[^\]]*\]", re.IGNORECASE),
        "role_bracket_spoofing",
    ),
    (
        re.compile(r"</?SYS>>", re.IGNORECASE),
        "llama_sys_delimiter_spoofing",
    ),
    (
        re.compile(r"###\s*(?:System|Instruction):", re.IGNORECASE),
        "markdown_role_spoofing",
    ),
    # 3. Data exfiltration and malicious markup
    (
        re.compile(r"!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=(?:key|token|secret|password|history|context)", re.IGNORECASE),
        "image_exfiltration_beacon",
    ),
    (
        re.compile(r"<script\b|javascript:", re.IGNORECASE),
        "script_payload",
    ),
]


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

# Reasons that only make sense for retrieved documents. A person asking about web markup in a
# question is not attacking the model, so these are not applied to user input.
_DOCUMENT_ONLY_REASONS = frozenset({"script_payload", "image_exfiltration_beacon"})


def normalize_for_scan(text: str) -> str:
    """Fold look-alike characters, drop invisible ones, and collapse whitespace before pattern matching.

    Without this, "ignore  previous\u200b instructions" or full-width letters slip past the patterns.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    return re.sub(r"\s+", " ", folded)


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
