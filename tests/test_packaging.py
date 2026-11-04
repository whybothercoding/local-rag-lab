"""Tests for package layout, pyproject.toml compliance, and CLI console scripts.

Verifies:
1. Valid PEP 517 / PEP 518 / PEP 621 declarative metadata in pyproject.toml.
2. Every console script entry point maps to an importable, callable entry point.
3. Every declared py-module exists on disk and imports cleanly.
4. Pytest options are correctly defined in pyproject.toml without deprecation warnings.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

from rag_lab.config import ROOT


def test_pyproject_toml_structure_and_metadata():
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist in the repository root"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # 1. Build system
    assert "build-system" in data
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=61.0" in data["build-system"]["requires"]

    # 2. Project metadata
    project = data.get("project", {})
    assert project.get("name") == "local-rag-lab"
    assert project.get("version") == "0.1.0"
    assert project.get("requires-python") == ">=3.11"
    deps = project.get("dependencies", [])
    assert len(deps) > 10
    assert any(d.startswith("langgraph-checkpoint-postgres") for d in deps), (
        "langgraph-checkpoint-postgres must be declared in pyproject.toml dependencies"
    )

    # Verify requirements.txt also includes langgraph-checkpoint-postgres
    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "langgraph-checkpoint-postgres" in req_text, (
        "langgraph-checkpoint-postgres must be declared in requirements.txt"
    )

    # 3. Tool configs
    assert "tool" in data
    assert "pytest" in data["tool"]
    assert "ini_options" in data["tool"]["pytest"]
    assert data["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_console_script_entrypoints_are_callable():
    pyproject_path = ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    scripts = data.get("project", {}).get("scripts", {})
    assert len(scripts) >= 5, "Expected standard console scripts to be configured"

    for script_name, target in scripts.items():
        module_name, func_name = target.split(":")
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        assert callable(func), f"Entry point {script_name} -> {target} must be callable"


def test_package_modules_import_and_ship_their_data():
    """Every module under rag_lab/ imports cleanly; the password denylist ships as package data."""
    import pkgutil

    import rag_lab

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == ["rag_lab*"]
    assert {"common_passwords.txt", "injection_patterns.tsv"} <= set(data["tool"]["setuptools"]["package-data"]["rag_lab.security"])
    assert "data/*.json" in data["tool"]["setuptools"]["package-data"]["rag_lab.lang"]

    # Modules that need an optional package (an extra in pyproject.toml): they must import where it is installed and may
    # fail to import, with exactly that package missing, where it is not.
    optional = {"rag_lab.storage.qdrant_store": "qdrant_client"}
    assert set(optional.values()) <= {"qdrant_client"} and "qdrant" in data["project"]["optional-dependencies"]

    names = [m.name for m in pkgutil.walk_packages(rag_lab.__path__, "rag_lab.")]
    assert len(names) >= 30
    for name in names:
        # The Streamlit page runs only inside a Streamlit session (see the root ui.py launcher).
        if name == "rag_lab.ui.streamlit_app":
            continue
        try:
            assert importlib.import_module(name) is not None
        except ModuleNotFoundError as exc:
            assert optional.get(name) == exc.name, f"{name} failed to import: {exc}"


def test_no_flat_modules_are_left_at_the_repository_root():
    """Only the Streamlit launcher stays at the root; a stray copy would shadow the package module."""
    assert sorted(p.name for p in ROOT.glob("*.py")) == ["ui.py"]


def test_console_scripts_execute_via_cli():
    import subprocess

    pyproject_path = ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    scripts = data.get("project", {}).get("scripts", {})
    venv_bin = ROOT / "venv" / "bin"

    for script_name in scripts:
        executable = venv_bin / script_name
        if not executable.is_file():
            continue
        res = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
        assert res.returncode == 0, f"Script {script_name} --help failed: {res.stderr}"


def test_tool_configurations_in_pyproject():
    pyproject_path = ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    tool = data.get("tool", {})
    assert "ruff" in tool
    assert tool["ruff"]["line-length"] == 100
    assert "mypy" in tool
    assert tool["mypy"]["python_version"] == "3.11"


def test_declared_langgraph_floor_has_room_for_the_retry_loops():
    """LangGraph < 1.1 stopped graphs after 25 steps; the worst retry path needs more (see test_retries)."""
    import re

    for manifest in ("pyproject.toml", "requirements.txt"):
        text = (Path(__file__).resolve().parent.parent / manifest).read_text(encoding="utf-8")
        floor = re.search(r'^\s*"?langgraph>=(\d+)\.(\d+)', text, re.MULTILINE)
        assert floor, manifest
        assert (int(floor.group(1)), int(floor.group(2))) >= (1, 1), manifest


def test_runtime_data_defaults_resolve_to_the_repository_root(monkeypatch):
    """Moving code into rag_lab/ must not move data: keys, policy, stores, and models stay at the root.

    A default inside the package would silently create fresh keys there, so the real policy would fail
    its integrity check and existing audit and checkpoint data could not be decrypted.
    """
    import rag_lab
    from rag_lab.config import DEFAULT_SQLITE_PATH, Settings

    repo = Path(__file__).resolve().parent.parent
    assert rag_lab.ROOT == repo
    path_fields = ("chroma_path", "bm25_dir", "uploads_dir", "knowledge_base_dir", "audit_db_path", "cache_db_path",
                   "access_policy_path", "access_key_path", "data_key_path", "anchor_path", "backup_dir", "models_dir")
    for field in path_fields + ("db_uri",):
        monkeypatch.delenv(field.upper(), raising=False)
    # A fresh instance, never a reload: every module holds the one shared settings object.
    defaults = Settings.from_env()
    paths = {field: getattr(defaults, field) for field in path_fields}
    paths["db_uri"] = defaults.db_uri.removeprefix("sqlite:///")
    paths["DEFAULT_SQLITE_PATH"] = DEFAULT_SQLITE_PATH
    for name, raw in paths.items():
        path = Path(raw).resolve()
        assert path.is_relative_to(repo), name
        assert not path.is_relative_to(repo / "rag_lab"), f"{name} points inside the package: {path}"


def test_rag_lab_home_overrides_the_root(monkeypatch, tmp_path):
    import importlib

    import rag_lab

    monkeypatch.setenv("RAG_LAB_HOME", str(tmp_path))
    try:
        assert importlib.reload(rag_lab).ROOT == tmp_path.resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(rag_lab)


def test_package_import_switches_off_model_hub_and_telemetry():
    import subprocess
    import sys

    code = "import os, rag_lab; print(os.environ['HF_HUB_OFFLINE'], os.environ['ANONYMIZED_TELEMETRY'])"
    env = {"PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=Path(__file__).resolve().parent.parent)
    assert out.stdout.split() == ["1", "False"], out.stderr


def test_password_denylist_ships_with_the_package():
    from rag_lab.security.access import _DENYLIST_PATH, _common_passwords

    assert _DENYLIST_PATH.is_file()
    assert len(_common_passwords()) >= 30  # only entries of at least MIN_PASSWORD_LENGTH matter


def test_settings_are_read_at_call_time(monkeypatch, tmp_path):
    """Overriding one settings field redirects every module that uses it; no module keeps a copy."""
    import ast

    from rag_lab.config import LEGACY_NAMES, settings
    from rag_lab.retrieval.sparse_search import get_fts_db_path

    monkeypatch.setattr(settings, "bm25_dir", str(tmp_path))
    assert get_fts_db_path("ws").startswith(str(tmp_path))

    # No module may import a former config constant or bind one at module level.
    for path in (Path(__file__).resolve().parent.parent / "rag_lab").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rag_lab.config":
                stale = [a.name for a in node.names if a.name in LEGACY_NAMES]
                assert not stale, f"{path.name} imports {stale}; read settings.<field> at call time instead"
