#!/usr/bin/env python3
"""Offline evaluation: retrieval quality, answer quality, and access-control behavior.

Builds its own index from eval/corpus/ under eval/out/ and runs the real LangGraph pipeline.
Every store is redirected into eval/out/ before the project imports load, so the evaluation
never reads or writes production data.

The default LLM is the deterministic in-process synthesizer, so results are reproducible and
need no network or model daemon. Set LLM_PROVIDER to score another backend.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
OUT_DIR = EVAL_DIR / "out"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.json"

# Redirect every store before config/app are imported. Setting these after import has no effect.
shutil.rmtree(OUT_DIR, ignore_errors=True)
os.environ["CHROMA_PATH"] = str(OUT_DIR / "chroma")
os.environ["BM25_DIR"] = str(OUT_DIR / "bm25")
os.environ["UPLOADS_DIR"] = str(OUT_DIR / "uploads")
os.environ["AUDIT_DB_PATH"] = str(OUT_DIR / "audit.db")
os.environ["ACCESS_POLICY_PATH"] = str(OUT_DIR / "access_policy.json")
os.environ["DB_URI"] = f"sqlite:///{OUT_DIR / 'unused_checkpoints.db'}"
os.environ.setdefault("LLM_PROVIDER", "deterministic")

from langgraph.checkpoint.memory import MemorySaver

from access import save_policy
from app import workflow
from eval_metrics import (
    check_thresholds,
    cited_sources,
    context_precision,
    context_recall,
    keyword_recall,
    summarize,
)
from ingest import ingest_workspace

# Pass/fail gates. Each one sits just below the score measured when this suite was first run.
# Abstention is reported but not gated: the pipeline does not yet refuse out-of-scope questions reliably.
DEFAULT_THRESHOLDS = {
    "retrieval_recall": 0.9,
    "keyword_recall": 0.8,
    "groundedness": 0.5,
    "security_accuracy": 1.0,
}


def seed_policy() -> None:
    """Evaluation accounts need no passwords; they exist only to exercise authorization."""
    save_policy({
        "version": 1,
        "users": {
            "eval": {"password_hash": "", "admin": False, "grants": {"engineering": ["query"], "finance": ["query"]}},
            "eval-outsider": {"password_hash": "", "admin": False, "grants": {}},
        },
    })


def build_indices() -> None:
    for workspace_dir in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        result = ingest_workspace(workspace_dir.name, str(workspace_dir))
        if not result.ok:
            raise SystemExit(f"Indexing failed for '{workspace_dir.name}': {result.error}")


def run_case(engine, case: dict) -> dict:
    thread_id = f"eval-{case['id']}"
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "workspace_id": case["workspace"],
        "session_id": thread_id,
        "principal": case.get("principal", "eval"),
        "question": case["question"],
        "original_question": case["question"],
        "sanitized_question": "",
        "documents": [],
        "citations": [],
        "generation": "",
        "intent": "",
        "cache_hit": False,
        "is_safe": True,
        "grade_status": "",
        "hallucination_status": "",
        "relevance_status": "",
        "retry_count": 0,
        "history": [],
        "telemetry": {},
    }

    started = time.perf_counter()
    for _ in engine.stream(inputs, config=config):
        pass
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    values = engine.get_state(config).values
    answer = values.get("generation", "")
    citations = values.get("citations", [])
    sources = cited_sources(citations)
    expect = case.get("expect", "answer")

    result = {
        "id": case["id"],
        "workspace": case["workspace"],
        "expect": expect,
        "question": case["question"],
        "answer": answer,
        "citations": citations,
        "latency_ms": latency_ms,
        "abstained": "could not find" in answer.lower(),
    }
    if expect == "denied":
        result["passed"] = "ACCESS DENIED" in answer
    elif expect == "blocked":
        result["passed"] = values.get("is_safe") is False and "SECURITY VIOLATION" in answer
    elif expect == "abstain":
        result["passed"] = result["abstained"]
    else:
        expected_sources = case.get("expected_sources", [])
        keywords = case.get("expected_keywords", [])
        result["context_recall"] = context_recall(sources, expected_sources)
        result["context_precision"] = context_precision(sources, expected_sources)
        result["keyword_recall"] = keyword_recall(answer, keywords)
        result["groundedness"] = values.get("telemetry", {}).get("groundedness_ratio")
        result["passed"] = result["context_recall"] == 1.0 and result["keyword_recall"] == 1.0
    return result


def print_table(results: list[dict], summary: dict) -> None:
    print(f"\n{'case':<24}{'expect':<9}{'recall':>8}{'prec':>7}{'keywd':>7}{'ground':>8}{'pass':>6}")
    print("-" * 69)
    for r in results:
        def fmt(value):
            return f"{value:.2f}" if isinstance(value, float) else "-"
        print(
            f"{r['id']:<24}{r['expect']:<9}{fmt(r.get('context_recall')):>8}"
            f"{fmt(r.get('context_precision')):>7}{fmt(r.get('keyword_recall')):>7}"
            f"{fmt(r.get('groundedness')):>8}{'yes' if r['passed'] else 'NO':>6}"
        )
    print("-" * 69)
    for key, value in summary.items():
        print(f"{key:<20} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline retrieval, answer, and access-control evaluation.")
    for metric, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--min-{metric.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()
    thresholds = {metric: getattr(args, f"min_{metric}") for metric in DEFAULT_THRESHOLDS}

    seed_policy()
    build_indices()
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    engine = workflow.compile(checkpointer=MemorySaver())
    results = [run_case(engine, case) for case in cases]
    summary = summarize(results)
    failures = check_thresholds(summary, thresholds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "thresholds": thresholds, "failures": failures, "results": results}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_table(results, summary)
    if failures:
        print("\nFAILED thresholds:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll thresholds met. Report: eval/out/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
