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


def keyword_recall(answer: str, keywords: list[str | list[str]]) -> float:
    """Share of the expected key facts that appear in the answer (case-insensitive).

    A fact given as a list matches if any alternative appears, so one case accepts an answer in the
    document's language or the question's ("montag" or "monday").
    """
    if not keywords:
        return 1.0
    text = answer.lower()

    def found(keyword: str | list[str]) -> bool:
        options = [keyword] if isinstance(keyword, str) else keyword
        return any(option.lower() in text for option in options)

    return sum(1 for keyword in keywords if found(keyword)) / len(keywords)


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["expect"] == "answer"]
    security = [r for r in results if r["expect"] in ("denied", "blocked")]
    cache = [r for r in results if r["expect"] in ("fresh", "cached")]
    abstain = [r for r in results if r["expect"] == "abstain"]
    judged = [r for r in answerable if r.get("judge_supported") is not None]
    latencies = [r["latency_ms"] for r in results]
    return {
        "cases": len(results),
        "answer_rate": mean([0.0 if r["abstained"] else 1.0 for r in answerable]),
        "retrieval_recall": mean([r.get("context_recall") for r in answerable]),
        "context_precision": mean([r.get("context_precision") for r in answerable]),
        "keyword_recall": mean([r.get("keyword_recall") for r in answerable]),
        "groundedness": mean([r.get("groundedness") for r in answerable]),
        "abstention_accuracy": mean([1.0 if r["passed"] else 0.0 for r in abstain]),
        "security_accuracy": mean([1.0 if r["passed"] else 0.0 for r in security]),
        "cache_accuracy": mean([1.0 if r["passed"] else 0.0 for r in cache]),
        "judge_support": mean([1.0 if r["judge_supported"] else 0.0 for r in judged]),
        "degraded_cases": sum(1 for r in results if r.get("degraded")),
        "median_latency_ms": round(median(latencies), 1) if latencies else None,
    }


def check_thresholds(summary: dict, thresholds: dict[str, float]) -> list[str]:
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric)
        if value is None and metric == "cache_accuracy":
            continue  # a suite without cache cases has nothing to gate there
        if value is None or value < minimum:
            failures.append(f"{metric} = {value} (required >= {minimum})")
    return failures
