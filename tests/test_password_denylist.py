"""The common-password list: large enough to matter at the minimum length, sourced and licensed, and fast."""
import time
from pathlib import Path

import pytest

from rag_lab.security import access
from rag_lab.security.access import MIN_PASSWORD_LENGTH, check_password_strength
from conftest import TEST_PASSWORD

LIST = Path(access._DENYLIST_PATH)


@pytest.fixture(scope="module")
def entries() -> list[str]:
    return [line for line in LIST.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def test_the_list_is_big_enough_to_matter_at_the_minimum_length(entries):
    """Passwords under 12 characters are refused by length alone, so only 12+ character entries count. A top-10k
    list has about two dozen of them; this one has tens of thousands."""
    assert len(entries) > 40_000
    assert all(len(e) >= MIN_PASSWORD_LENGTH for e in entries), "shorter entries can never match"
    assert all(e == e.lower() and e == e.strip() for e in entries), "lookups lowercase the candidate"
    assert len(set(entries)) == len(entries)


def test_source_and_licence_are_recorded():
    header = "\n".join(line for line in LIST.read_text(encoding="utf-8").splitlines() if line.startswith("#"))
    assert "danielmiessler/SecLists" in header and "35d74839e66f44d116e694369d86a5bf07aaed6f" in header
    assert "xato-net-10-million-passwords-1000000.txt" in header and "MIT" in header
    licence = LIST.with_name("common_passwords.LICENSE").read_text()
    assert licence.startswith("MIT License") and "Daniel Miessler" in licence and "WITHOUT WARRANTY" in licence
    pyproject = (LIST.parents[2] / "pyproject.toml").read_text()
    assert "common_passwords.LICENSE" in pyproject, "the notice ships with the data"


@pytest.mark.parametrize("password", ["Passwordpassword", "ILOVEYOU1234", "Welcome12345"])
def test_common_passwords_are_refused_in_any_case(password):
    with pytest.raises(ValueError, match="too common"):
        check_password_strength(password)


def test_entries_from_the_new_list_are_refused(entries):
    for sample in (entries[0], entries[len(entries) // 2], entries[-1]):
        if sample.lower() not in access._common_passwords():
            continue
        with pytest.raises(ValueError, match="too common|account name"):
            check_password_strength(sample.upper())


def test_the_passwords_the_test_suite_relies_on_are_not_on_it():
    for password in (TEST_PASSWORD, "new-strong-password-999", "another-valid-password-123"):
        check_password_strength(password)


def test_loading_and_looking_up_are_fast():
    access._common_passwords.cache_clear()
    start = time.perf_counter()
    words = access._common_passwords()
    load = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20_000):
        TEST_PASSWORD in words
    lookups = time.perf_counter() - start
    assert load < 0.25, f"first call took {load:.3f}s"
    assert lookups < 0.1, f"20k lookups took {lookups:.3f}s"
