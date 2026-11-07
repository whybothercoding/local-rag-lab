#!/usr/bin/env python3
"""Offline evaluation: retrieval, answers, abstention, access control, cache safety, and conversations.

Builds its own index from eval/corpus/ under eval/out/ and runs the real LangGraph pipeline. Every
store is redirected into eval/out/ before the project imports load, so the evaluation never reads or
writes production data.

Cases carry a `suite` (en, multilingual, conversation) and a `lang` (question language). Each suite
has its own gates, measured on its held-out test split; the calibration split is for choosing
thresholds. A suite whose gates are None is measured and reported but not gated yet (see SUITE_GATES).

Default mode uses the deterministic in-process synthesizer, so results are reproducible and need no
model daemon (this is what CI gates). `--llm` runs the same cases against the local Ollama model and
also gates an LLM judge that checks every answer is supported by its retrieved context.

--calibrate prints the retrieval signals per calibration case, for choosing thresholds.
--judge adds the judge to a deterministic run (reported, not gated).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from rag_lab import ROOT  # the package root only: no project module may load before the redirect below

EVAL_DIR = ROOT / "eval"
# EVAL_OUT_DIR lets tests and parallel runs keep their stores and report apart from eval/out/.
OUT_DIR = Path(os.environ.get("EVAL_OUT_DIR") or EVAL_DIR / "out")
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.json"

from rag_lab.ops.eval_metrics import (  # pure functions: importing them loads no settings
    check_thresholds,
    cited_sources,
    context_precision,
    context_recall,
    keyword_recall,
    summarize,
)


def redirect_stores(out_dir: Path, llm_mode: bool) -> None:
    """Point every store at out_dir (wiping it) before any project module that reads settings is imported.

    settings is built from the environment on its first import, so this must run first. Importing this
    module has no side effects; only main() calls it.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ["CHROMA_PATH"] = str(out_dir / "chroma")
    os.environ["BM25_DIR"] = str(out_dir / "bm25")
    os.environ["UPLOADS_DIR"] = str(out_dir / "uploads")
    os.environ["AUDIT_DB_PATH"] = str(out_dir / "audit.db")
    os.environ["CACHE_DB_PATH"] = str(out_dir / "cache.db")
    os.environ["ACCESS_POLICY_PATH"] = str(out_dir / "access_policy.json")
    os.environ["ACCESS_KEY_PATH"] = str(out_dir / "access.key")
    os.environ["DATA_KEY_PATH"] = str(out_dir / "data.key")
    os.environ["ANCHOR_PATH"] = str(out_dir / "audit_anchors.jsonl")
    # Throwaway keys belong in out_dir, never in the OS keychain.
    os.environ["KEY_STORE"] = "file"
    os.environ["KEY_ENVELOPE"] = ""
    os.environ["AUDIT_ANCHOR_COPY_DIR"] = ""
    # The evaluation measures the embedded defaults; a .env that enables a scaling option must not redirect it.
    # To evaluate an adapter on purpose, say so in the shell: `VECTOR_STORE=qdrant make eval` (docs/scaling.md). Even
    # then it gets its own namespace, forced here: an exported real prefix or schema must never receive the eval corpus,
    # whose workspaces (engineering, finance, hr...) would overwrite real ones.
    os.environ.setdefault("VECTOR_STORE", "chroma")
    os.environ["QDRANT_PREFIX"] = "eval_"
    os.environ["PGVECTOR_SCHEMA"] = "eval"
    os.environ["CACHE_BACKEND"] = "sqlite"
    os.environ["CACHE_DB_URI"] = ""
    os.environ["DB_URI"] = f"sqlite:///{out_dir / 'unused_checkpoints.db'}"
    if llm_mode:
        os.environ["LLM_PROVIDER"] = "ollama"
    else:
        os.environ.setdefault("LLM_PROVIDER", "deterministic")
    if "rag_lab.config" in sys.modules:
        raise RuntimeError("rag_lab.config was imported before the eval stores were redirected")


# Pass/fail gates per suite, measured on that suite's held-out test split. Thresholds were chosen from
# the calibration split. None would mean measured and reported, not gated.
DEFAULT_THRESHOLDS = {
    "answer_rate": 0.9,
    "retrieval_recall": 0.9,
    "keyword_recall": 0.8,
    "groundedness": 0.5,
    "abstention_accuracy": 0.8,
    "security_accuracy": 1.0,
    "cache_accuracy": 1.0,
    "chitchat_accuracy": 1.0,
    "entailment_rate": 0.9,
}
# German, French, Greek, and cross-lingual questions. Keyword recall is lower than English because the
# deterministic synthesizer quotes the document's language while some facts are checked in either language.
MULTILINGUAL_THRESHOLDS = {
    "answer_rate": 0.9,
    "retrieval_recall": 0.9,
    "keyword_recall": 0.75,
    "groundedness": 0.5,
    "abstention_accuracy": 0.9,
    "security_accuracy": 1.0,
    "cache_accuracy": 1.0,
    "chitchat_accuracy": 1.0,
    "entailment_rate": 0.9,
}
# Follow-ups that only make sense with the earlier turn. Keyword recall is the extractive synthesizer's
# limit (it cannot infer "business class" from "longer flights"); retrieval finds every expected source.
CONVERSATION_THRESHOLDS = {
    "answer_rate": 0.9,
    "retrieval_recall": 0.9,
    "keyword_recall": 0.7,
    "groundedness": 0.5,
    "entailment_rate": 0.9,
}
SUITE_GATES: dict[str, dict[str, float] | None] = {
    "en": DEFAULT_THRESHOLDS,
    "multilingual": MULTILINGUAL_THRESHOLDS,
    "conversation": CONVERSATION_THRESHOLDS,
}
# Added to every gated suite in --llm mode: share of answers the judge finds supported by their context.
LLM_JUDGE_MIN = float(os.getenv("EVAL_LLM_JUDGE_MIN", "0.8"))

JUDGE_PROMPT = (
    "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
    "Is every factual claim in the answer supported by the context? The context and the answer may be in "
    "different languages. Reply with exactly one word: yes or no."
)

WORKSPACES = ("engineering", "finance", "hr", "zurich", "geneve", "athens")


def seed_policy() -> None:
    """Evaluation accounts need no passwords; they exist only to exercise authorization and clearance."""
    from rag_lab.security.access import save_policy

    save_policy({
        "version": 3,
        "defaults": {"role": "none"},
        "groups": {},
        "users": {
            "eval": {
                "password_hash": "", "clearance": "public",
                "uid": "eval0001", "token_version": 0,
                "grants": {ws: ["query", "auto_approve"] for ws in WORKSPACES},
            },
            "eval-secret": {
                "password_hash": "", "clearance": "secret",
                "uid": "eval0002", "token_version": 0,
                "grants": {"engineering": ["query", "auto_approve"], "zurich": ["query", "auto_approve"]},
            },
            "eval-outsider": {
                "password_hash": "", "clearance": "public",
                "uid": "eval0003", "token_version": 0,
                "grants": {},
            },
        },
    })


def build_indices() -> None:
    from rag_lab.ingestion.ingest import ingest_workspace

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


def _inputs(case: dict, question: str, thread_id: str) -> dict:
    from rag_lab.pipeline.entry import prepare_pipeline_input
    from rag_lab.security.access import issue_token

    return prepare_pipeline_input(
        workspace_id=case["workspace"],
        session_id=thread_id,
        principal_token=issue_token(case.get("principal", "eval")),
        question=question,
    )


def _ask(engine, case: dict, question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    for _ in engine.stream(_inputs(case, question, thread_id), config=config):
        pass
    return dict(engine.get_state(config).values)


def run_case(engine, case: dict, judge_llm=None) -> dict:
    from rag_lab.retrieval.semantic_cache import cache_stats, clear_cache

    clear_cache()
    thread_id = f"eval-{case['id']}"
    # Cache cases: earlier questions in their own threads, so only the shared cache links them.
    for i, earlier in enumerate(case.get("setup", [])):
        _ask(engine, case, earlier, f"{thread_id}-setup-{i}")
    setup_cached = cache_stats()["total_entries"] > 0
    # Conversation cases: earlier turns in the same thread as the follow-up. A turn that was not released
    # (a real model's answer sent to review) gives the follow-up no context: that case is inconclusive.
    context_released = True
    for earlier in case.get("history", []):
        turn = _ask(engine, case, earlier, thread_id)
        last = (turn.get("history") or [{}])[-1]
        pending = engine.get_state({"configurable": {"thread_id": thread_id}}).next
        context_released = context_released and not pending and isinstance(last, dict) and bool(last.get("released"))
    if not context_released:
        return {"id": case["id"], "split": case.get("split", "test"), "suite": case.get("suite", "en"),
                "lang": case.get("lang", "en"), "workspace": case["workspace"], "expect": case.get("expect", "answer"),
                "question": case["question"], "answer": "", "citations": [], "latency_ms": 0.0, "abstained": False,
                "cache_hit": False, "best_similarity": None, "best_coverage": None, "best_rerank_score": None,
                "relevant": None, "degraded": [], "judge_supported": None, "passed": None,
                "note": "earlier turn not released"}

    started = time.perf_counter()
    values = _ask(engine, case, case["question"], thread_id)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    telemetry = values.get("telemetry", {})
    answer = values.get("generation", "")
    citations = values.get("citations", [])
    sources = cited_sources(citations)
    expect = case.get("expect", "answer")
    cache_hit = bool(telemetry.get("cache_hit"))

    result = {
        "id": case["id"],
        "split": case.get("split", "test"),
        "suite": case.get("suite", "en"),
        "lang": case.get("lang", "en"),
        "workspace": case["workspace"],
        "expect": expect,
        "question": case["question"],
        "standalone_question": values.get("original_question", case["question"]),
        "answer": answer,
        "citations": citations,
        "documents": values.get("documents", []),
        "latency_ms": latency_ms,
        "abstained": "could not find" in answer.lower() or telemetry.get("abstained") is True,
        "cache_hit": cache_hit,
        "best_similarity": telemetry.get("best_similarity"),
        "best_coverage": telemetry.get("best_coverage"),
        "best_rerank_score": telemetry.get("best_rerank_score"),
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
    elif expect == "chitchat":
        result["passed"] = values.get("intent") == "chitchat"
    elif expect == "cached":
        result["passed"] = cache_hit
    elif expect == "fresh":
        # Only meaningful when the earlier answer was cached; otherwise nothing could have been reused, and
        # the case is inconclusive (None). A real model's answer may go to review instead of the cache.
        result["setup_cached"] = setup_cached
        result["passed"] = (not cache_hit) if setup_cached else None
    else:
        expected_sources = case.get("expected_sources", [])
        keywords = case.get("expected_keywords", [])
        result["context_recall"] = context_recall(sources, expected_sources)
        result["context_precision"] = context_precision(sources, expected_sources)
        result["keyword_recall"] = keyword_recall(answer, keywords)
        result["groundedness"] = telemetry.get("groundedness_ratio")
        result["nli_supported"] = telemetry.get("nli_supported")
        result["passed"] = (
            result["context_recall"] == 1.0 and result["keyword_recall"] == 1.0 and not result["abstained"]
        )
        if judge_llm is not None and not result["abstained"]:
            result["judge_supported"] = judge_support(judge_llm, values.get("documents", []), answer)
    return result


def print_calibration(results: list[dict]) -> None:
    print(f"\n{'case':<28}{'lang':<5}{'expect':<9}{'similarity':>11}{'coverage':>10}{'relevant':>10}  note")
    print("-" * 81)
    for r in results:
        sim = "-" if r["best_similarity"] is None else f"{r['best_similarity']:.3f}"
        cov = "-" if r["best_coverage"] is None else f"{r['best_coverage']:.2f}"
        note = "degraded" if r["degraded"] else ""
        print(f"{r['id']:<28}{r['lang']:<5}{r['expect']:<9}{sim:>11}{cov:>10}{str(r['relevant']):>10}  {note}")


def print_table(results: list[dict], summary: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'case':<28}{'lang':<5}{'expect':<9}{'recall':>8}{'prec':>7}{'keywd':>7}{'ground':>8}{'pass':>6}")
    print("-" * 78)
    for r in results:
        def fmt(value):
            return f"{value:.2f}" if isinstance(value, float) else "-"
        print(
            f"{r['id']:<28}{r['lang']:<5}{r['expect']:<9}{fmt(r.get('context_recall')):>8}"
            f"{fmt(r.get('context_precision')):>7}{fmt(r.get('keyword_recall')):>7}"
            f"{fmt(r.get('groundedness')):>8}{('n/a' if r['passed'] is None else 'yes' if r['passed'] else 'NO'):>6}"
        )
    print("-" * 78)
    for key, value in summary.items():
        print(f"{key:<22} {value}")


def print_language_breakdown(results: list[dict]) -> None:
    """Pass rate per question language across every suite's test split."""
    print(f"\n{'language':<10}{'cases':>7}{'passed':>8}{'rate':>7}")
    for lang in sorted({r["lang"] for r in results}):
        rows = [r for r in results if r["lang"] == lang and r["passed"] is not None]
        passed = sum(1 for r in rows if r["passed"])
        if not rows:
            # Every case of this language was inconclusive (e.g. an earlier turn never released).
            print(f"{lang:<10}{0:>7}{0:>8}{'-':>7}  (no conclusive cases)")
            continue
        print(f"{lang:<10}{len(rows):>7}{passed:>8}{passed / len(rows):>7.2f}")


def judge_report(path: Path) -> int:
    """Add judge verdicts to a saved report and print judge support per suite (test split)."""
    from langchain_ollama import ChatOllama

    model = os.getenv("EVAL_JUDGE_MODEL")
    if not model:
        print("--judge-only needs EVAL_JUDGE_MODEL: a model from another family than the one that answered.")
        return 2
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    judge = ChatOllama(model=model, base_url=base_url, temperature=0.0, reasoning=False)
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["results"]:
        if row["expect"] == "answer" and row.get("passed") is not None and not row["abstained"] and row.get("documents"):
            row["judge_supported"] = judge_support(judge, row["documents"], row["answer"])
    report["judge_model"] = model
    for suite in report["suites"]:
        test = [r for r in report["results"] if r["suite"] == suite and r["split"] == "test"]
        report["suites"][suite]["test"] = summarize(test)
        print(f"{suite:<13} judge_support={report['suites'][suite]['test']['judge_support']}")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def suite_gates(suite: str, overrides: dict[str, float], llm_mode: bool) -> dict[str, float] | None:
    gates = SUITE_GATES.get(suite)
    if gates is None:
        return None
    gates = {**gates, **(overrides if suite == "en" else {})}
    if llm_mode:
        gates["judge_support"] = LLM_JUDGE_MIN
    elif "cache_accuracy" in gates:
        # Deterministic answers are always released, so every cache case must actually test the cache.
        gates["cache_conclusive_rate"] = 1.0
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline retrieval, answer, abstention, access, and cache evaluation.")
    for metric, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--min-{metric.replace('_', '-')}", type=float, default=default,
                            help=f"gate for the en suite (default {default})")
    parser.add_argument("--calibrate", action="store_true", help="Print retrieval signals for the calibration split")
    parser.add_argument("--judge", action="store_true", help="Also score answer support with the local Ollama model")
    parser.add_argument("--llm", action="store_true",
                        help="Answer with the local Ollama model and gate the LLM judge (needs 'ollama serve')")
    parser.add_argument("--judge-only", type=Path, metavar="REPORT",
                        help="Judge the answers saved in REPORT with EVAL_JUDGE_MODEL, in one pass (the judge "
                             "loads once instead of swapping with the answering model on every case)")
    parser.add_argument("--suite", action="append", choices=sorted(SUITE_GATES),
                        help="Run only these suites (repeatable); default all")
    args = parser.parse_args()
    overrides = {metric: getattr(args, f"min_{metric}") for metric in DEFAULT_THRESHOLDS}

    if args.judge_only:
        return judge_report(args.judge_only)

    redirect_stores(OUT_DIR, args.llm)
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.memory import MemorySaver

    from rag_lab.pipeline.graph import workflow
    from rag_lab.pipeline.llm_provider import get_chat_llm, is_ollama_alive

    judge_llm: BaseChatModel | None = None
    if args.llm or args.judge:
        if not is_ollama_alive():
            message = "Ollama is not reachable at its configured URL. Start it with 'ollama serve'."
            if args.llm:
                print(f"--llm needs a model: {message}")
                return 2
            print(f"Judge skipped: {message}")
        else:
            # A different judge model avoids the answering model grading itself (EVAL_JUDGE_MODEL).
            judge_model = os.getenv("EVAL_JUDGE_MODEL")
            if judge_model:
                from langchain_ollama import ChatOllama

                from rag_lab.config import settings

                judge_llm = ChatOllama(model=judge_model, base_url=settings.ollama_base_url, temperature=0.0)
            else:
                judge_llm = get_chat_llm("ollama")

    seed_policy()
    build_indices()
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.suite:
        cases = [c for c in cases if c.get("suite", "en") in args.suite]

    engine = workflow.compile(checkpointer=MemorySaver())
    results = [run_case(engine, case, judge_llm) for case in cases]

    if args.calibrate:
        print_calibration([r for r in results if r["split"] == "calibration"])
        return 0

    report: dict = {"mode": "llm" if args.llm else "deterministic", "suites": {}, "results": results}
    all_failures: list[str] = []
    for suite in [s for s in SUITE_GATES if any(r["suite"] == s for r in results)]:
        rows = [r for r in results if r["suite"] == suite]
        calibration = [r for r in rows if r["split"] == "calibration"]
        test = [r for r in rows if r["split"] == "test"]
        cal_summary, test_summary = summarize(calibration), summarize(test)
        gates = suite_gates(suite, overrides, args.llm)
        failures = check_thresholds(test_summary, gates) if gates is not None else []
        all_failures += [f"[{suite}] {f}" for f in failures]
        report["suites"][suite] = {
            "calibration": cal_summary, "test": test_summary, "thresholds": gates, "failures": failures,
        }
        status = "gated" if gates is not None else "reported, not gated yet"
        print_table(test, test_summary, f"SUITE {suite.upper()}: TEST SPLIT ({status})")

    print_language_breakdown([r for r in results if r["split"] == "test"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_failures:
        print("\nFAILED thresholds:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    print("\nAll gated suites met their thresholds. Report: eval/out/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
