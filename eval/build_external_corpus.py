#!/usr/bin/env python3
"""Build eval/external/: a small, fixed slice of XQuAD (English, German, Greek) for `make eval-external`.

XQuAD (Artetxe, Ruder, Yogatama 2019) is 240 Wikipedia paragraphs and 1,190 questions from SQuAD v1.1 development
set, professionally translated and entirely parallel across languages. Licence: CC BY-SA 4.0. It has no French.
The source files are not committed; this script fetches them from a pinned commit, verifies their SHA-256,
and writes the slice. Run it once (needs the network); everything after that is offline.

Selection rule, fixed before any evaluation run and never revisited:
  1. Take articles in file order and drop the three whose subject is an individual person (BIOGRAPHIES).
  2. From each remaining article keep its first PARAGRAPHS paragraphs, and from each paragraph its first
     QUESTIONS questions, all in file order.
  3. Every kept paragraph becomes one document per language (workspace xquad_<lang>); every kept question becomes
     one case per language (same-language documents) plus, for German and Greek, one cross-lingual case against
     the English documents (expected answer: the English span).
No thresholds, prompts or pipeline settings were looked at or changed to produce this slice. Remaining text is public
encyclopedic content: it names public figures incidentally and contains no data about private individuals.

    python eval/build_external_corpus.py            # fetch, verify, write eval/external/
    python eval/build_external_corpus.py --source DIR   # use already downloaded xquad.<lang>.json files from DIR
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "external"
COMMIT = "7d30520c717524000f0d9d2f9c10a069acd9d285"  # google-deepmind/xquad
RAW = f"https://raw.githubusercontent.com/google-deepmind/xquad/{COMMIT}/"
SHA256 = {
    "xquad.en.json": "e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29",
    "xquad.de.json": "990b5d746746ed65ed4702ea5f35f99ffa4e2f1c390c07d003642acd937916f9",
    "xquad.el.json": "821cf0f88e73fa258fd2f548b19b6ec39f7025059e16f6f9fc8cd797c9c3663e",
    "CC-BY-SA4.0.txt": "25ddafc153e4cfdafd70729574ef1b44348e1814709e57680d55347038d3f92c",
}
LANGS = ("en", "de", "el")
BIOGRAPHIES = {"Nikola_Tesla", "Martin_Luther", "Genghis_Khan"}
PARAGRAPHS = 2
QUESTIONS = 2


def load(name: str, source: Path | None) -> bytes:
    data = (source / name).read_bytes() if source else urllib.request.urlopen(RAW + name, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        sys.exit(f"{name}: SHA-256 does not match the pinned commit {COMMIT}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="directory that already holds the pinned source files")
    args = parser.parse_args()

    data = {lang: json.loads(load(f"xquad.{lang}.json", args.source))["data"] for lang in LANGS}
    licence = load("CC-BY-SA4.0.txt", args.source)

    shutil.rmtree(ROOT / "corpus", ignore_errors=True)
    cases: list[dict] = []
    for a, article in enumerate(data["en"]):
        if article["title"] in BIOGRAPHIES:
            continue
        for p, paragraph in enumerate(article["paragraphs"][:PARAGRAPHS]):
            name = f"a{a:02d}_p{p}.md"
            for lang in LANGS:
                target = data[lang][a]["paragraphs"][p]
                assert target["qas"][0]["id"] == paragraph["qas"][0]["id"], "files are not parallel"
                folder = ROOT / "corpus" / f"xquad_{lang}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(target["context"].strip() + "\n", encoding="utf-8")
            for q, qa in enumerate(paragraph["qas"][:QUESTIONS]):
                own = {lang: data[lang][a]["paragraphs"][p]["qas"][q] for lang in LANGS}
                assert len({own[lang]["id"] for lang in LANGS}) == 1
                for lang in LANGS:
                    cases.append({
                        "id": f"xq-{lang}-{lang}-{qa['id'][-8:]}", "suite": "external", "split": "test",
                        "lang": lang, "workspace": f"xquad_{lang}", "question": own[lang]["question"],
                        "expected_sources": [name], "expected_keywords": [own[lang]["answers"][0]["text"]],
                    })
                for lang in ("de", "el"):
                    cases.append({
                        "id": f"xq-{lang}-en-{qa['id'][-8:]}", "suite": "external", "split": "test",
                        "lang": lang, "workspace": "xquad_en", "question": own[lang]["question"],
                        "expected_sources": [name], "expected_keywords": [own["en"]["answers"][0]["text"]],
                    })
    (ROOT / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "LICENSE-CC-BY-SA-4.0.txt").write_bytes(licence)
    print(f"{len(cases)} cases, {sum(1 for _ in (ROOT / 'corpus').rglob('*.md'))} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
