"""Plain-text files are decoded in-process; the result must be exactly what the sandboxed child returns."""
import pytest

from rag_lab.config import settings
from rag_lab.ingestion import parsers
from rag_lab.ingestion.parsers import PLAIN_SUFFIXES, ParseError, _extract_in_child, extract_text_from_file

CONTENTS = {
    "plain": b"Hello plain text world\n",
    "crlf": b"line one\r\nline two\r\n\r\nline four\r",
    "bom": b"\xef\xbb\xbfStarts with a byte order mark\n",
    "invalid-utf8": b"caf\xe9 and \xff\xfe broken bytes\n",
    "nul": b"before\x00after\n",
    "unicode-separators": "para sep more\x0bvt\x1cfs\x85nel\n".encode(),
    "greek-german": "Η άδεια γάμου είναι 5 εργάσιμες ημέρες. Größe & Maße.\n".encode(),
    "long-line": (b"word " * 40_000) + b"\n",
    "whitespace-only": b"  \n\t \n",
    "empty": b"",
    "frontmatter": b"---\nclearance: secret\n---\n# Title\nBody\n",
}


@pytest.mark.parametrize("suffix", sorted(PLAIN_SUFFIXES))
def test_every_plain_suffix_is_read_in_process(tmp_path, suffix, monkeypatch):
    path = tmp_path / f"note{suffix}"
    path.write_bytes(b"Some text\n")
    monkeypatch.setattr(parsers.subprocess, "run", lambda *a, **k: pytest.fail("a plain-text file must not start a child"))
    assert extract_text_from_file(path) == "Some text"


@pytest.mark.parametrize("name", list(CONTENTS))
def test_the_in_process_text_is_exactly_the_childs(tmp_path, name):
    path = tmp_path / "doc.md"
    path.write_bytes(CONTENTS[name])
    assert extract_text_from_file(path) == _extract_in_child(path)


def test_the_child_receives_the_parse_timeout_so_its_cpu_cap_matches(tmp_path, monkeypatch):
    """Without forwarding PARSE_TIMEOUT_SEC the child always caps at the 60 s default while the parent
    waits for the configured wall-clock timeout: a slow parse would die at 65 s CPU with no reason."""
    seen: dict = {}
    real = parsers.subprocess.run

    def fake_run(*a, **k):
        seen.update(k.get("env") or {})
        return real(*a, **k)

    monkeypatch.setattr(parsers.subprocess, "run", fake_run)
    monkeypatch.setattr(parsers, "PARSE_TIMEOUT_SEC", 300)
    path = tmp_path / "doc.json"
    path.write_text('{"k": "v"}')
    assert extract_text_from_file(path)
    assert seen.get("PARSE_TIMEOUT_SEC") == "300"


def test_everything_else_still_runs_in_the_sandboxed_child(tmp_path, monkeypatch):
    calls = []
    real = parsers.subprocess.run
    monkeypatch.setattr(parsers.subprocess, "run", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    for name, body in (("a.csv", "x,y\n1,2\n"), ("b.json", '{"k": "v"}'), ("c.html", "<p>hi</p>")):
        (tmp_path / name).write_text(body)
        assert extract_text_from_file(tmp_path / name)
    assert len(calls) == 3 and all("parser_worker" in " ".join(c) for c in calls)


def test_a_plain_file_over_the_size_limit_goes_to_the_child_which_has_memory_caps(tmp_path, monkeypatch):
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(settings, "max_source_bytes", 50)
    seen = []
    monkeypatch.setattr(parsers, "_extract_in_child", lambda p: seen.append(p) or "from child")
    assert extract_text_from_file(path) == "from child" and seen == [path]


def test_a_file_that_grows_past_the_limit_while_being_read_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "grows.txt"
    path.write_bytes(b"y" * 10)
    monkeypatch.setattr(settings, "max_source_bytes", 10)
    path.write_bytes(b"y" * 5000)  # after the size check would have passed: simulate with a direct call
    with pytest.raises(ParseError, match="larger than 10 bytes"):
        parsers._extract_plain_bounded(path, 10)


def test_unreadable_or_missing_files_are_a_parse_error_like_before(tmp_path):
    with pytest.raises(ParseError):
        extract_text_from_file(tmp_path / "missing.md")
    locked = tmp_path / "locked.md"
    locked.write_text("secret")
    locked.chmod(0)
    try:
        with pytest.raises(ParseError):
            extract_text_from_file(locked)
    finally:
        locked.chmod(0o600)


def test_unsupported_types_are_still_empty(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00")
    assert extract_text_from_file(tmp_path / "x.bin") == ""
