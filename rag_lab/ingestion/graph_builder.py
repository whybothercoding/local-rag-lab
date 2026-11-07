"""Knowledge Graph entity and triplet builder for workspace ingestion.

Extracts entities (People, Roles, Departments, Policies, Systems) and directed
relationships from document chunks during `make sync`.
"""
from __future__ import annotations

import re
from typing import Any

from rag_lab.lang import fold
from rag_lab.retrieval.graph_store import KnowledgeGraph


# Patterns for extracting enterprise relationships across English, German, French, and Greek.
_RELATION_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    # English
    (re.compile(r"([A-Z][a-zA-Z.\s]{2,30}?)\s+(?:is|serves as|holds the title of|holds the role of)\s+(?:the\s+)?([A-Z][a-zA-Z.\s]{2,40})"), "PERSON", "HOLDS_ROLE", "ROLE"),
    (re.compile(r"([A-Z][a-zA-Z0-9_.\s]{2,40}?)\s+(?:requires|needs|must be|may only be|can only be)\s+(?:the\s+)?(?:approval|sign-off|authorization|authorized|approved)\s+(?:from|by)\s+(?:the\s+)?([A-Z][a-zA-Z.\s]{2,40})"), "POLICY", "REQUIRES_APPROVAL", "ROLE"),
    (re.compile(r"([A-Z][a-zA-Z0-9_.\s]{2,40}?)\s+(?:supersedes|overrides|replaces)\s+([A-Z][a-zA-Z0-9_.\s]{2,40})"), "POLICY", "SUPERSEDES", "POLICY"),
    (re.compile(r"([A-Z][a-zA-Z0-9_.\s]{2,40}?)\s+(?:applies to|governs|references|pertains to)\s+([A-Z][a-zA-Z0-9_.\s]{2,40})"), "POLICY", "APPLIES_TO", "ENTITY"),
    (re.compile(r"([A-Z][a-zA-Z0-9_.\s]{2,40}?)\s+(?:depends on|relies on)\s+([A-Z][a-zA-Z0-9_.\s]{2,40})"), "SERVICE", "DEPENDS_ON", "SERVICE"),

    # German
    (re.compile(r"([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß.\s]{2,30}?)\s+(?:ist|fungiert als)\s+(?:der|die|das)?\s*([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß.\s]{2,40})"), "PERSON", "HOLDS_ROLE", "ROLE"),
    (re.compile(r"([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß0-9_.\s]{2,40}?)\s+(?:erfordert|benötigt|darf nur)\s+(?:die\s+)?(?:Genehmigung|Freigabe|genehmigt werden)\s+durch\s+([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß.\s]{2,40})"), "POLICY", "REQUIRES_APPROVAL", "ROLE"),
    (re.compile(r"([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß0-9_.\s]{2,40}?)\s+ersetzt\s+([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß0-9_.\s]{2,40})"), "POLICY", "SUPERSEDES", "POLICY"),

    # French
    (re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ.\s]{2,30}?)\s+est\s+(?:le|la)?\s*([A-Za-zÀ-ÖØ-öø-ÿ.\s]{2,40})"), "PERSON", "HOLDS_ROLE", "ROLE"),
    (re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ0-9_.\s]{2,40}?)\s+(?:nécessite|requiert|doit être)\s+(?:l'approbation|autorisé)\s+(?:de|par)\s+([A-Za-zÀ-ÖØ-öø-ÿ.\s]{2,40})"), "POLICY", "REQUIRES_APPROVAL", "ROLE"),

    # Greek
    (re.compile(r"([\u0370-\u03ff\u1f00-\u1fff.\s]{2,30}?)\s+(?:είναι|κατέχει τη θέση του|κατέχει τη θέση της)\s+([\u0370-\u03ff\u1f00-\u1fff.\s]{2,40})", re.IGNORECASE), "PERSON", "HOLDS_ROLE", "ROLE"),
    (re.compile(r"([\u0370-\u03ff\u1f00-\u1fff0-9_.\s]{2,40}?)\s+(?:απαιτεί|χρειάζεται|εγκρίνεται)\s+(?:έγκριση\s+από|μόνο\s+από)\s+([\u0370-\u03ff\u1f00-\u1fff.\s]{2,40})", re.IGNORECASE), "POLICY", "REQUIRES_APPROVAL", "ROLE"),
]

_COMMON_WORDS = frozenset({
    "all", "this", "that", "these", "those", "each", "every", "some", "none",
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "der", "die", "das", "ein", "eine", "einer", "und", "oder", "wenn",
    "le", "la", "les", "un", "une", "des", "et", "ou", "si",
    "ο", "η", "το", "οι", "τα", "και", "ή", "αν", "όταν", "όλα", "κάθε",
})


def _clean_name(name: str) -> str:
    cleaned = name.strip().strip(".,;:\"'()[]{}")
    # Drop leading articles
    words_list = cleaned.split()
    if words_list and fold(words_list[0]) in _COMMON_WORDS:
        cleaned = " ".join(words_list[1:])
    return cleaned.strip()


def extract_triplets_from_text(text: str, chunk_id: str) -> list[dict[str, Any]]:
    """Extract relation triplets (source, predicate, target) from text chunk."""
    triplets: list[dict[str, Any]] = []
    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue

        for pattern, src_type, pred, tgt_type in _RELATION_PATTERNS:
            for match in pattern.finditer(line_clean):
                groups = match.groups()
                if len(groups) == 2:
                    src_name = _clean_name(groups[0])
                    tgt_name = _clean_name(groups[1])
                    if len(src_name) >= 3 and len(tgt_name) >= 3:
                        if fold(src_name) not in _COMMON_WORDS and fold(tgt_name) not in _COMMON_WORDS:
                            triplets.append({
                                "source": src_name,
                                "source_type": src_type,
                                "predicate": pred,
                                "target": tgt_name,
                                "target_type": tgt_type,
                                "chunk_id": chunk_id,
                            })
    return triplets


def build_workspace_graph(workspace_id: str, chunks: list[dict[str, Any]]) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a list of chunk dicts (each having 'id' and 'text')."""
    kg = KnowledgeGraph(workspace_id=workspace_id)

    for chunk in chunks:
        cid = str(chunk.get("id") or "")
        text = str(chunk.get("text") or chunk.get("page_content") or "")
        if not text:
            continue

        triplets = extract_triplets_from_text(text, cid)
        for t in triplets:
            kg.add_relationship(
                source_name=t["source"],
                target_name=t["target"],
                predicate=t["predicate"],
                source_chunk=t["chunk_id"],
                source_type=t["source_type"],
                target_type=t["target_type"],
            )

    return kg
