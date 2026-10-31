"""Scoring functions for the offline evaluation. Pure functions with no I/O, so they are unit-testable."""
import re
from statistics import median

_CITATION = re.compile(r"^\[[^\]]+\]\s*(.*)$")


def cited_sources(citations: list[str]) -> list[str]:
    """Source file for each citation shaped '[chunk_id] source'."""
    sources = []
    for citation in citations:
        match = _CITATION.match(citation.strip())
        sources.append((match.group(1) if match else citation).strip())
    return sources


def context_recall(cited: list[str], expected: list[str]) -> float:
    """Share of the expected source files that were retrieved."""
    if not expected:
        return 1.0
    found = set(cited)
    return sum(1 for source in expected if source in found) / len(expected)


def context_precision(cited: list[str], expected: list[str]) -> float:
    """Share of the retrieved chunks that come from an expected source."""
    if not cited:
        return 0.0
    wanted = set(expected)
    return sum(1 for source in cited if source in wanted) / len(cited)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Share of the expected key facts that appear in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(1 for keyword in keywords if keyword.lower() in text) / len(keywords)


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def summarize(results: list[dict]) -> dict:
    answered = [r for r in results if r["expect"] == "answer"]
    security = [r for r in results if r["expect"] in ("denied", "blocked")]
    abstain = [r for r in results if r["expect"] == "abstain"]
    latencies = [r["latency_ms"] for r in results]
    return {
        "cases": len(results),
        "retrieval_recall": mean([r.get("context_recall") for r in answered]),
        "context_precision": mean([r.get("context_precision") for r in answered]),
        "keyword_recall": mean([r.get("keyword_recall") for r in answered]),
        "groundedness": mean([r.get("groundedness") for r in answered]),
        "security_accuracy": mean([1.0 if r["passed"] else 0.0 for r in security]),
        "abstention_rate": mean([1.0 if r["abstained"] else 0.0 for r in abstain]),
        "median_latency_ms": round(median(latencies), 1) if latencies else None,
    }


def check_thresholds(summary: dict, thresholds: dict[str, float]) -> list[str]:
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric)
        if value is None or value < minimum:
            failures.append(f"{metric} = {value} (required >= {minimum})")
    return failures
