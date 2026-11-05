"""Score a labelled battery of (claim, source, supported?) pairs with candidate NLI models, on this machine.

Run from the repository root. Prints a table and writes <out>.json so two machines can be compared.
usage: python nli_battery.py <cache_dir> <out.json> [variant ...]
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
WANTED = sys.argv[3:]

VARIANTS = {
    "int8-current": ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c", "onnx/model_quantized.onnx"),
    "fp32-current": ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c", "onnx/model.onnx"),
    "fp32-2mil7": ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", None, "onnx/model.onnx"),
    "fp32-minilm-l6": ("onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX", None, "onnx/model.onnx"),
}
names = WANTED or list(VARIANTS)

# 1. downloads first: importing rag_lab sets HF_HUB_OFFLINE=1
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402

paths = {}
for name in names:
    repo, revision, weights = VARIANTS[name]
    if revision is None:
        revision = HfApi().model_info(repo).sha  # the current main, recorded in the output
    files = {f: hf_hub_download(repo, f, revision=revision, cache_dir=str(CACHE)) for f in ("config.json", "tokenizer.json", weights)}
    paths[name] = (files, revision, weights)
    print(f"downloaded {name}: {repo}@{revision[:10]} {weights} {os.path.getsize(files[weights])/1e6:.0f} MB", flush=True)

# 2. scoring with the repository's own claim-versus-sources logic
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from rag_lab.config import settings  # noqa: E402
from rag_lab.retrieval import nli  # noqa: E402

DOC = lambda p: open(f"eval/corpus/{p}", encoding="utf-8").read()  # noqa: E731
TRAVEL, EXPENSES, DEPLOY, ROLLBACK, LEAVE = DOC("finance/travel.md"), DOC("finance/expenses.md"), DOC("engineering/deployments.md"), DOC("engineering/rollback.md"), DOC("hr/leave.md")
CONGES, KUND, HOME, EXODA, ADEIES, TELE = DOC("geneve/conges.md"), DOC("zurich/kuendigung.md"), DOC("zurich/homeoffice.md"), DOC("athens/exoda.md"), DOC("athens/adeies.md"), DOC("geneve/teletravail.md")

# (label, language, claim, source). True = the source states it.
BATTERY = [
    (True, "en", "Economy class is the default for flights under six hours.", TRAVEL),
    (True, "en", "Flights shorter than six hours are booked in economy.", TRAVEL),
    (True, "de", "Für Flüge unter sechs Stunden ist Economy der Standard.", TRAVEL),
    (True, "en", "Business class needs prior approval from the department director.", TRAVEL),
    (True, "en", "Hotel rates above 180 EUR per night need finance review before booking.", TRAVEL),
    (True, "en", "Standard production deployments happen on Tuesday and Thursday.", DEPLOY),
    (True, "en", "Emergency hotfixes need written sign-off from the VP of Engineering.", DEPLOY),
    (True, "en", "The rollback SLA is under 5 minutes.", ROLLBACK),
    (True, "en", "Employees accrue 25 days of annual leave per year.", LEAVE),
    (True, "en", "All business expense receipts above $50 require manager sign-off.", EXPENSES),
    (True, "en", "Expense reports must be submitted within 30 days of purchase.", EXPENSES),
    (True, "fr", "Chaque collaborateur a droit à 25 jours de congés payés par an.", CONGES),
    (True, "fr", "Le télétravail est autorisé jusqu'à 2 jours par semaine.", TELE),
    (True, "fr", "L'entreprise fournit un écran et un casque pour le télétravail.", TELE),
    (True, "de", "Nach der Probezeit gilt eine Kündigungsfrist von 2 Monaten auf das Monatsende.", KUND),
    (True, "de", "Während der Probezeit beträgt die Kündigungsfrist 7 Tage.", KUND),
    (True, "de", "Am Montag ist Homeoffice nicht erlaubt.", HOME),
    (True, "de", "Die Firma erstattet 45 CHF pro Monat für das Homeoffice.", HOME),
    (True, "el", "Οι εργαζόμενοι δικαιούνται 22 ημέρες άδειας τον πρώτο χρόνο.", ADEIES),
    (True, "el", "Η άδεια γάμου είναι 5 εργάσιμες ημέρες.", ADEIES),
    (True, "el", "Οι αποδείξεις εξόδων υποβάλλονται εντός 30 ημερών.", EXODA),
    (True, "el", "Οι πτήσεις κάτω από 4 ώρες κλείνονται σε οικονομική θέση.", EXODA),
    (False, "en", "Business class is the default for flights under six hours.", TRAVEL),
    (False, "en", "Business class requires approval from the department director and the CEO's signature.", TRAVEL),
    (False, "en", "Hotel rates above 250 EUR per night need finance review.", TRAVEL),
    (False, "en", "Standard production deployments happen on Monday and Friday.", DEPLOY),
    (False, "en", "The rollback SLA is under 30 minutes.", ROLLBACK),
    (False, "en", "Employees accrue 30 days of annual leave per year.", LEAVE),
    (False, "en", "Receipts for business expenses above $50 require manager sign-off before reimbursement.", EXPENSES),
    (False, "en", "Expense reports must be submitted within 90 days of purchase.", EXPENSES),
    (False, "fr", "Chaque collaborateur a droit à 30 jours de congés payés par an.", CONGES),
    (False, "fr", "Le télétravail est autorisé jusqu'à 4 jours par semaine.", TELE),
    (False, "fr", "Le télétravail depuis l'étranger est autorisé sans accord.", TELE),
    (False, "de", "Nach der Probezeit gilt eine Kündigungsfrist von 6 Monaten.", KUND),
    (False, "de", "Während der Probezeit beträgt die Kündigungsfrist 3 Monate.", KUND),
    (False, "de", "Am Freitag ist Homeoffice nicht erlaubt.", HOME),
    (False, "de", "Die Firma erstattet 100 CHF pro Monat für das Homeoffice.", HOME),
    (False, "el", "Οι εργαζόμενοι δικαιούνται 30 ημέρες άδειας τον πρώτο χρόνο.", ADEIES),
    (False, "el", "Η άδεια γάμου είναι 10 εργάσιμες ημέρες.", ADEIES),
    (False, "el", "Οι αποδείξεις εξόδων υποβάλλονται εντός 30 ημερών.", ADEIES),   # right fact, wrong source
    (False, "el", "Το ημερήσιο επίδομα για ταξίδια εσωτερικού είναι 80 ευρώ.", EXODA),
    (False, "el", "Οι πτήσεις κάτω από 4 ώρες κλείνονται σε επαγγελματική θέση.", EXODA),
]


def build(files, weights):
    config = json.loads(Path(files["config.json"]).read_text())
    entail = next(int(i) for i, n in config["id2label"].items() if n.lower() == "entailment")
    tok = Tokenizer.from_file(files["tokenizer.json"])
    tok.enable_truncation(max_length=512)
    return ort.InferenceSession(files[weights], providers=["CPUExecutionProvider"]), tok, entail


def auc(scores):
    pos = [s for (t, s) in scores if t]
    neg = [s for (t, s) in scores if not t]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


results = {"platform": f"{platform.machine()} {platform.system()}", "ort": ort.__version__, "variants": {}}
try:
    results["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except OSError:
    results["cpu"] = platform.processor() or platform.machine()
print(results["platform"], "| ORT", ort.__version__, "|", results["cpu"], flush=True)

for name in names:
    files, revision, weights = paths[name]
    session, tok, entail = build(files, weights)
    nli._SESSIONS[(settings.nli_model, settings.models_dir)] = (session, tok, entail)
    started = time.perf_counter()
    scores = [(label, nli.sentence_support(claim, [source])) for label, _, claim, source in BATTERY]
    elapsed = time.perf_counter() - started
    values = [s for _, s in scores]
    t = min(s for lab, s in scores if lab)
    f = max(s for lab, s in scores if not lab)
    gate = settings.groundedness_min_entailment
    wrong = [(BATTERY[i][1], BATTERY[i][2][:48], round(s, 2), lab) for i, (lab, s) in enumerate(scores) if (s >= gate) != lab]
    print(f"{name:<16} AUC {auc(scores):.3f} | lowest true {t:.2f} | highest false {f:.2f} | wrong at gate {gate}: {len(wrong)}/{len(BATTERY)} | {elapsed/len(BATTERY)*1000:.0f} ms/claim", flush=True)
    for w in wrong:
        print("     wrong:", w)
    results["variants"][name] = {"revision": revision, "weights": weights, "scores": values, "auc": auc(scores), "ms_per_claim": elapsed / len(BATTERY) * 1000}

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)
