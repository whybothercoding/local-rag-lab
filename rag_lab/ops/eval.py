#!/usr/bin/env python3
"""Offline evaluation: retrieval, answer, abstention, and access-control behavior.

Builds its own index from eval/corpus/ under eval/out/ and runs the real LangGraph pipeline.
Every store is redirected into eval/out/ before the project imports load, so the evaluation
never reads or writes production data.

Cases are split into a calibration split, used to choose thresholds, and a held-out test split,
which the gates measure. The default LLM is the deterministic in-process synthesizer, so results
are reproducible and need no network or model daemon. Set LLM_PROVIDER to score another backend.

--calibrate prints the retrieval signals per calibration case, for choosing thresholds.
--judge additionally asks the local Ollama model whether each answer is supported by its context.
The judge is reported, never gated.
"""
import argparse
import json
import os
import shutil
import sys
import time

from rag_lab import ROOT  # the package root only: no project module may load before the redirect below

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
os.environ["CACHE_DB_PATH"] = str(OUT_DIR / "cache.db")
os.environ["ACCESS_POLICY_PATH"] = str(OUT_DIR / "access_policy.json")
os.environ["ACCESS_KEY_PATH"] = str(OUT_DIR / "access.key")
os.environ["DATA_KEY_PATH"] = str(OUT_DIR / "data.key")
os.environ["ANCHOR_PATH"] = str(OUT_DIR / "audit_anchors.jsonl")
os.environ["DB_URI"] = f"sqlite:///{OUT_DIR / 'unused_checkpoints.db'}"
os.environ.setdefault("LLM_PROVIDER", "deterministic")

from langgraph.checkpoint.memory import MemorySaver

from rag_lab.security.access import issue_token, save_policy
from rag_lab.pipeline.graph import workflow
from rag_lab.ops.eval_metrics import (
    check_thresholds,
    cited_sources,
    context_precision,
    context_recall,
    keyword_recall,
    summarize,
)
from rag_lab.ingestion.ingest import ingest_workspace
from rag_lab.pipeline.llm_provider import get_chat_llm, is_ollama_alive
from rag_lab.retrieval.semantic_cache import clear_cache

# Pass/fail gates, measured on the held-out test split only. Thresholds were chosen from the
# calibration split. Judge support is reported but not gated.
DEFAULT_THRESHOLDS = {
    "answer_rate": 0.9,
    "retrieval_recall": 0.9,
    "keyword_recall": 0.8,
    "groundedness": 0.5,
    "abstention_accuracy": 0.8,
    "security_accuracy": 1.0,
}

JUDGE_PROMPT = (
    "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
    "Is every factual claim in the answer supported by the context? Reply with exactly one word: yes or no."
)


def seed_policy() -> None:
    """Evaluation accounts need no passwords; they exist only to exercise authorization and clearance."""
    save_policy({
        "version": 2,
        "defaults": {"permissions": []},
        "users": {
            "eval": {
                "password_hash": "", "admin": False, "clearance": "public",
                "uid": "eval0001", "token_version": 0,
                "grants": {
                    "engineering": ["query", "auto_approve"],
                    "finance": ["query", "auto_approve"],
                    "hr": ["query", "auto_approve"],
                },
            },
            "eval-secret": {
                "password_hash": "", "admin": False, "clearance": "secret",
                "uid": "eval0002", "token_version": 0,
                "grants": {"engineering": ["query", "auto_approve"]},
            },
            "eval-outsider": {
                "password_hash": "", "admin": False, "clearance": "public",
                "uid": "eval0003", "token_version": 0,
                "grants": {},
            },
        },
    })


def build_indices() -> None:
    for workspace_dir in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        result = ingest_workspace(workspace_dir.name, str(workspace_dir))
        if not result.ok:
            raise SystemExit(f"Indexing failed for '{workspace_dir.name}': {result.error}")


def judge_support(llm, documents: list[str], answer: str) -> bool | None:
    prompt = JUDGE_PROMPT.format(context="\n\n---\n\n".join(documents), answer=answer)
    try:
        reply = str(llm.invoke(prompt).content).strip().lower()
    except Exception:
        return None
    return reply.startswith("yes")


def run_case(engine, case: dict, judge_llm=None) -> dict:
    clear_cache()
    thread_id = f"eval-{case['id']}"
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "workspace_id": case["workspace"],
        "session_id": thread_id,
        "principal_token": issue_token(case.get("principal", "eval")),
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
        "generation_retry_count": 0,
        "grader_feedback": "",
        "history": [],
        "telemetry": {},
        "degraded": [],
    }

    started = time.perf_counter()
    for _ in engine.stream(inputs, config=config):
        pass
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    values = engine.get_state(config).values
    telemetry = values.get("telemetry", {})
    answer = values.get("generation", "")
    citations = values.get("citations", [])
    sources = cited_sources(citations)
    expect = case.get("expect", "answer")

    result = {
        "id": case["id"],
        "split": case.get("split", "test"),
        "workspace": case["workspace"],
        "expect": expect,
        "question": case["question"],
        "answer": answer,
        "citations": citations,
        "latency_ms": latency_ms,
        "abstained": "could not find" in answer.lower(),
        "best_similarity": telemetry.get("best_similarity"),
        "best_coverage": telemetry.get("best_coverage"),
        "relevant": telemetry.get("relevant"),
        "degraded": values.get("degraded", []),
        "judge_supported": None,
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
        result["groundedness"] = telemetry.get("groundedness_ratio")
        result["passed"] = (
            result["context_recall"] == 1.0 and result["keyword_recall"] == 1.0 and not result["abstained"]
        )
        if judge_llm is not None and not result["abstained"]:
            result["judge_supported"] = judge_support(judge_llm, values.get("documents", []), answer)
    return result


def print_calibration(results: list[dict]) -> None:
    print(f"\n{'case':<26}{'expect':<9}{'similarity':>11}{'coverage':>10}{'relevant':>10}  note")
    print("-" * 74)
    for r in results:
        sim = "-" if r["best_similarity"] is None else f"{r['best_similarity']:.3f}"
        cov = "-" if r["best_coverage"] is None else f"{r['best_coverage']:.2f}"
        note = "degraded" if r["degraded"] else ""
        print(f"{r['id']:<26}{r['expect']:<9}{sim:>11}{cov:>10}{str(r['relevant']):>10}  {note}")


def print_table(results: list[dict], summary: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'case':<26}{'expect':<9}{'recall':>8}{'prec':>7}{'keywd':>7}{'ground':>8}{'pass':>6}")
    print("-" * 71)
    for r in results:
        def fmt(value):
            return f"{value:.2f}" if isinstance(value, float) else "-"
        print(
            f"{r['id']:<26}{r['expect']:<9}{fmt(r.get('context_recall')):>8}"
            f"{fmt(r.get('context_precision')):>7}{fmt(r.get('keyword_recall')):>7}"
            f"{fmt(r.get('groundedness')):>8}{'yes' if r['passed'] else 'NO':>6}"
        )
    print("-" * 71)
    for key, value in summary.items():
        print(f"{key:<22} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline retrieval, answer, abstention, and access evaluation.")
    for metric, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--min-{metric.replace('_', '-')}", type=float, default=default)
    parser.add_argument("--calibrate", action="store_true", help="Print retrieval signals for the calibration split")
    parser.add_argument("--judge", action="store_true", help="Also score answer support with the local Ollama model")
    args = parser.parse_args()
    thresholds = {metric: getattr(args, f"min_{metric}") for metric in DEFAULT_THRESHOLDS}

    seed_policy()
    build_indices()
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    judge_llm = None
    if args.judge:
        if is_ollama_alive():
            judge_llm = get_chat_llm("ollama")
        else:
            print("Judge skipped: Ollama is not reachable at its configured URL.")

    engine = workflow.compile(checkpointer=MemorySaver())
    results = [run_case(engine, case, judge_llm) for case in cases]

    if args.calibrate:
        print_calibration([r for r in results if r["split"] == "calibration"])
        return 0

    calibration = [r for r in results if r["split"] == "calibration"]
    test = [r for r in results if r["split"] == "test"]
    cal_summary, test_summary = summarize(calibration), summarize(test)
    failures = check_thresholds(test_summary, thresholds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "calibration": cal_summary,
        "test": test_summary,
        "thresholds": thresholds,
        "failures": failures,
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_table(test, test_summary, "TEST SPLIT (gated)")
    print(f"\nCalibration split: {cal_summary['cases']} cases, not gated. See eval/out/report.json.")
    if failures:
        print("\nFAILED thresholds (test split):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll thresholds met on the test split. Report: eval/out/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
