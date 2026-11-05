"""Reranker logits for fixed (query, passage) pairs, to compare the int8 reranker across CPUs. Run from the repo root."""
import json
import platform
import sys
import time

from rag_lab.config import settings
from rag_lab.security.model_integrity import verify_model
from rag_lab.retrieval.reranker import score_pairs

DOC = lambda p: open(f"eval/corpus/{p}", encoding="utf-8").read()  # noqa: E731
D = {k: DOC(v) for k, v in dict(deploy="engineering/deployments.md", rollback="engineering/rollback.md", travel="finance/travel.md",
                                leave="hr/leave.md", expenses="finance/expenses.md", kuend="zurich/kuendigung.md",
                                home="zurich/homeoffice.md", tele="geneve/teletravail.md", conges="geneve/conges.md",
                                adeies="athens/adeies.md", exoda="athens/exoda.md").items()}
CASES = [  # query, relevant, distractors
    ("On which days are standard production deployments scheduled?", "deploy", ["rollback", "travel"]),
    ("How many days of annual leave do employees get?", "leave", ["expenses", "deploy"]),
    ("What is the rollback SLA?", "rollback", ["deploy", "leave"]),
    ("Who must approve business class travel?", "travel", ["expenses", "deploy"]),
    ("Wie lange ist die Kündigungsfrist nach der Probezeit?", "kuend", ["home", "travel"]),
    ("An wie vielen Tagen darf ich im Homeoffice arbeiten?", "home", ["kuend", "leave"]),
    ("Combien de jours de télétravail par semaine ?", "tele", ["conges", "expenses"]),
    ("Combien de jours de congés payés ?", "conges", ["tele", "deploy"]),
    ("Πόσες ημέρες άδειας δικαιούνται οι εργαζόμενοι;", "adeies", ["exoda", "leave"]),
    ("Σε πόσες ημέρες υποβάλλονται οι αποδείξεις εξόδων;", "exoda", ["adeies", "expenses"]),
]

out = {"platform": f"{platform.machine()} {platform.system()}", "variants": {}}
print(out["platform"], flush=True)
for name in ("BAAI/bge-reranker-v2-m3-int8", "BAAI/bge-reranker-v2-m3"):
    try:
        verify_model(name, settings.models_dir)
    except Exception as exc:
        print(f"{name}: not present here ({type(exc).__name__})")
        continue
    started = time.perf_counter()
    scores = [score_pairs(q, [D[r]] + [D[x] for x in rest], model_name=name) for q, r, rest in CASES]
    elapsed = time.perf_counter() - started
    right = sum(1 for s in scores if s and s[0] == max(s))
    gap = min(s[0] - max(s[1:]) for s in scores)
    print(f"{name:<30} relevant ranked first {right}/{len(CASES)} | smallest margin {gap:.2f} logits | {elapsed/len(CASES)*1000:.0f} ms/query", flush=True)
    out["variants"][name] = [[round(x, 4) for x in s] for s in scores]
print("RERANK_JSON", json.dumps(out))
