"""PostgreSQL credentials come from .env: no password in code or compose, a loopback-only port, a safe .env bootstrap."""
import stat
from pathlib import Path

import pytest
import yaml

from rag_lab.config import postgres_uri_from_env
from rag_lab.ops import init_env

ROOT = Path(__file__).resolve().parent.parent


def test_there_is_no_default_password_in_code(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    assert postgres_uri_from_env() == ""


def test_the_uri_is_built_from_the_environment_and_quoted(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "svc user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/w:rd#1")
    monkeypatch.setenv("POSTGRES_DB", "cp")
    monkeypatch.setenv("POSTGRES_PORT", "5544")
    assert postgres_uri_from_env() == "postgresql://svc%20user:p%40ss%2Fw%3Ard%231@localhost:5544/cp"


def test_compose_has_no_password_and_listens_on_loopback_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["postgres"]
    password = service["environment"]["POSTGRES_PASSWORD"]
    assert password.startswith("${POSTGRES_PASSWORD:?"), "no literal and no default: compose refuses to start without it"
    assert all(p.startswith("127.0.0.1:") for p in service["ports"]), "the database is not offered to the LAN"
    assert "rag_password" not in (ROOT / "docker-compose.yml").read_text()


def test_no_password_is_committed_outside_ci():
    offenders = []
    for path in [*ROOT.glob("*.example"), ROOT / ".env.example", ROOT / "docker-compose.yml", ROOT / "Makefile",
                 ROOT / "README.md", *(ROOT / "rag_lab").rglob("*.py")]:
        if path.is_file() and "rag_password" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], "only the throwaway CI service container may carry a literal password"
    example = (ROOT / ".env.example").read_text()
    assert "\nPOSTGRES_PASSWORD=" not in example, ".env.example must not ship a live placeholder password"


def test_init_env_creates_a_private_env_with_a_random_password(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(init_env, "ROOT", tmp_path)
    assert init_env.main() == 0
    env = tmp_path / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    values = dict(line.split("=", 1) for line in env.read_text().splitlines() if line and not line.startswith("#"))
    assert values["POSTGRES_USER"] == "rag_user" and len(values["POSTGRES_PASSWORD"]) >= 30
    assert "DB_URI" not in values, "creating a database must not silently switch the app off SQLite"
    assert "make db-password" in capsys.readouterr().out

    other = tmp_path / "second"
    other.mkdir()
    monkeypatch.setattr(init_env, "ROOT", other)
    init_env.main()
    assert (other / ".env").read_text() != env.read_text(), "each install gets its own password"


def test_init_env_never_touches_an_existing_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(init_env, "ROOT", tmp_path)
    env = tmp_path / ".env"
    env.write_text("OLLAMA_MODEL=custom\n")
    env.chmod(0o640)
    before = env.stat()
    assert init_env.main() == 0
    assert env.read_text() == "OLLAMA_MODEL=custom\n" and stat.S_IMODE(env.stat().st_mode) == 0o640
    assert env.stat().st_mtime_ns == before.st_mtime_ns
    assert capsys.readouterr().out == ""


def test_the_postgres_tests_skip_without_credentials_and_fail_when_required(monkeypatch):
    """The fixture's message names the fix; it must not fall back to a baked-in password."""
    source = (ROOT / "tests" / "conftest.py").read_text()
    assert "postgres_uri_from_env" in source and "POSTGRES_DB_URI" not in source


@pytest.mark.parametrize("name", ["POSTGRES_PASSWORD"])
def test_the_gitignore_covers_env(name):
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()
