"""The scaling options are off by default, loopback-only, validated, documented, and cannot leak into tests."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from rag_lab.config import settings

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]
SCALING = ("qdrant", "pgvector", "vllm")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


def test_make_up_starts_only_the_checkpoint_database():
    assert {name for name, svc in SERVICES.items() if "profiles" not in svc} == {"postgres"}
    up = MAKEFILE.split("\nup:")[1].split("\n\n")[0]
    assert not any(name in up for name in (*SCALING, "caddy")) and "--profile" not in up


@pytest.mark.parametrize("name", [*SCALING, "caddy"])
def test_each_option_has_exactly_one_profile_named_after_it(name):
    assert SERVICES[name]["profiles"] == [name]


@pytest.mark.parametrize("name", SCALING)
def test_scaling_services_listen_on_loopback_only(name):
    ports = SERVICES[name]["ports"]
    assert ports and all(str(p).startswith("127.0.0.1:") for p in ports), ports


def test_images_are_named_and_the_moving_ones_are_documented():
    assert SERVICES["qdrant"]["image"] == "qdrant/qdrant:v1.20.0", "pinned: the client and the server move together"
    docs = (ROOT / "docs" / "scaling.md").read_text(encoding="utf-8")
    for floating in ("pgvector/pgvector:pg16", "vllm/vllm-openai:latest"):
        assert SERVICES[next(n for n, s in SERVICES.items() if s["image"] == floating)]["image"] == floating
        assert floating in docs and "pin a digest" in docs


def test_no_secret_is_written_into_compose():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "rag_password" not in text
    assert SERVICES["pgvector"]["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?"), "no default password"
    assert SERVICES["qdrant"]["environment"]["QDRANT__SERVICE__API_KEY"] == "${QDRANT_API_KEY:-}"
    assert SERVICES["vllm"]["environment"]["VLLM_API_KEY"] == "${VLLM_API_KEY:-}"


def test_vllm_asks_for_a_gpu_because_it_cannot_run_without_one():
    devices = SERVICES["vllm"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia" and "gpu" in devices[0]["capabilities"]
    docs = (ROOT / "docs" / "scaling.md").read_text(encoding="utf-8")
    assert "never run" in docs, "the docs must say this profile was validated, not run"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
@pytest.mark.parametrize("profile", [*SCALING, "caddy"])
def test_every_profile_is_a_valid_compose_configuration(profile):
    env = {**os.environ, "POSTGRES_PASSWORD": "not-a-real-password"}
    done = subprocess.run(["docker", "compose", "--profile", profile, "config", "--quiet"], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    if done.returncode != 0 and "Cannot connect to the Docker daemon" in done.stderr:
        pytest.skip("the docker daemon is not running")
    assert done.returncode == 0, done.stderr


def test_the_make_targets_exist_are_listed_and_check_their_argument():
    assert re.search(r"^\.PHONY:.*\bscale-up\b.*\bscale-down\b", MAKEFILE, re.MULTILINE)
    help_text = MAKEFILE.split("help:")[1].split("status:")[0]
    assert "make scale-up SERVICE=" in help_text and "make scale-down SERVICE=" in help_text
    for target in ("scale-up", "scale-down"):
        recipe = MAKEFILE.split(f"\n{target}:")[1].split("\n\n")[0]
        assert "qdrant|pgvector|vllm" in recipe and "--profile $(SERVICE)" in recipe


# --- defaults and isolation ----------------------------------------------------------------------------------------

def test_every_option_is_off_by_default():
    assert settings.vector_store == "chroma" and settings.cache_backend == "sqlite"
    assert settings.llm_provider == "deterministic", "tests run on the in-process synthesizer"
    assert not settings.qdrant_url and not settings.pgvector_uri and not settings.cache_db_uri


def test_the_eval_redirect_cannot_be_hijacked_by_an_env_file():
    source = (ROOT / "rag_lab" / "ops" / "eval.py").read_text(encoding="utf-8")
    assert 'os.environ["CACHE_BACKEND"] = "sqlite"' in source and 'os.environ["CACHE_DB_URI"] = ""' in source
    assert 'os.environ["QDRANT_PREFIX"] = "eval_"' in source and 'os.environ["PGVECTOR_SCHEMA"] = "eval"' in source, \
        "forced, not defaulted: an exported real prefix or schema must not receive the eval corpus"


# --- documentation does not drift from the code ------------------------------------------------------------------------------

SETTING_VARS = ("VECTOR_STORE", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_PREFIX", "PGVECTOR_URI", "PGVECTOR_SCHEMA",
                "CACHE_BACKEND", "CACHE_DB_URI", "LOCAL_LLM_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY",
                "LOCAL_LLM_TEMPLATE_KWARGS")


@pytest.mark.parametrize("variable", SETTING_VARS)
def test_every_scaling_variable_is_read_by_the_code_and_documented_twice(variable):
    assert f'"{variable}"' in (ROOT / "rag_lab" / "config.py").read_text(encoding="utf-8")
    assert f"{variable}" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert variable in (ROOT / "docs" / "scaling.md").read_text(encoding="utf-8")


def test_the_docs_state_what_does_not_scale():
    docs = (ROOT / "docs" / "scaling.md").read_text(encoding="utf-8")
    for local_file in ("bm25_indices/", "audit.db", "access_policy.json"):
        assert local_file in docs
    assert "same data key" in docs and "KEY_STORE=file" in docs
