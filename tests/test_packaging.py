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
import pytest

from config import ROOT


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


def test_declared_py_modules_exist_and_import():
    import ast

    pyproject_path = ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    modules = data.get("tool", {}).get("setuptools", {}).get("py-modules", [])
    assert len(modules) >= 20

    for mod_name in modules:
        py_file = ROOT / f"{mod_name}.py"
        assert py_file.is_file(), f"Declared module {mod_name}.py must exist at project root"

        # Verify syntactically valid python AST
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        assert tree is not None

        # Verify non-UI modules import cleanly (ui.py requires streamlit runtime context)
        if mod_name != "ui":
            mod = importlib.import_module(mod_name)
            assert mod is not None


def test_console_scripts_execute_via_cli():
    import subprocess
    import sys

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
