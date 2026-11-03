"""Long-lived checkpointers: one connection or pool per process, shared safely by concurrent sessions."""
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langgraph.types import Command

from app import workflow
from checkpointer import (
    close_checkpointer,
    fetch_checkpoint_threads,
    get_checkpointer,
    open_checkpointer,
    shared_pool,
)
from conftest import TEST_PASSWORD, token_for

UI = str(Path(__file__).resolve().parent.parent / "ui.py")


def _inputs(thread: str) -> dict:
    return {
        "workspace_id": "default",
        "session_id": thread,
        "principal_token": token_for("carol"),
        "question": "What is the rollback SLA?",
        "telemetry": {},
    }


def test_open_checkpointer_stays_open_until_closed(tmp_path):
    uri = f"sqlite:///{tmp_path / 'cp.db'}"
    saver = open_checkpointer(uri)
    engine = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t-open"}}
    engine.update_state(config, {"question": "q"}, as_node="guardrail")
    assert engine.get_state(config).values["question"] == "q"

    assert shared_pool(saver) is None, "a SQLite connection is never handed out as a pool"
    close_checkpointer(saver)
    close_checkpointer(saver)  # idempotent
    with pytest.raises(sqlite3.ProgrammingError):
        engine.get_state(config)


def test_get_checkpointer_still_closes_on_exit(tmp_path):
    with get_checkpointer(uri=f"sqlite:///{tmp_path / 'cp.db'}") as saver:
        engine = workflow.compile(checkpointer=saver)
    with pytest.raises(sqlite3.ProgrammingError):
        engine.get_state({"configurable": {"thread_id": "after-close"}})
    assert shared_pool(None) is None


def test_failed_setup_does_not_leak_the_connection(tmp_path, monkeypatch):
    import checkpointer

    closed = []

    real_connect = sqlite3.connect

    class Tracked:
        def __init__(self, *args, **kwargs):
            self._conn = real_connect(":memory:", check_same_thread=False)

        def close(self):
            closed.append(True)
            self._conn.close()

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def broken_saver(conn):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(checkpointer.sqlite3, "connect", lambda *a, **k: Tracked())
    monkeypatch.setattr(checkpointer, "_bound_sqlite_saver", broken_saver)
    with pytest.raises(RuntimeError, match="setup failed"):
        open_checkpointer(f"sqlite:///{tmp_path / 'cp.db'}")
    assert closed == [True]


def test_one_shared_sqlite_saver_serves_concurrent_sessions(tmp_path, rollback_index):
    """The UI shares one saver across browser sessions (threads). Each thread's sealed state stays its own."""
    uri = f"sqlite:///{tmp_path / 'shared.db'}"
    saver = open_checkpointer(uri)
    try:
        engine = workflow.compile(checkpointer=saver)
        threads = [f"ui-default-{i}" for i in range(8)]
        barrier = threading.Barrier(len(threads))

        def run(thread: str):
            barrier.wait()
            config = {"configurable": {"thread_id": thread}}
            for _ in engine.stream(_inputs(thread), config=config):
                pass
            return engine.get_state(config)

        with ThreadPoolExecutor(max_workers=len(threads)) as pool:
            states = list(pool.map(run, threads))

        assert all(s.next == ("human_approval",) for s in states)
        assert all(s.values["session_id"] == t for s, t in zip(states, threads))
        assert sorted(fetch_checkpoint_threads(uri=uri, prefix="ui-default-")) == sorted(threads)

        # Resuming one thread leaves the others paused.
        resume = {"decision": "approved", "approver_token": token_for("bob"), "reason": "ok"}
        first = {"configurable": {"thread_id": threads[0]}}
        for _ in engine.stream(Command(resume=resume), config=first):
            pass
        assert engine.get_state(first).next == ()
        assert engine.get_state({"configurable": {"thread_id": threads[1]}}).next == ("human_approval",)
    finally:
        close_checkpointer(saver)


def test_postgres_checkpointer_is_shared_and_pool_is_reused(postgres_uri, monkeypatch):
    import psycopg_pool

    saver = open_checkpointer(postgres_uri, max_pool_size=3)
    try:
        pool = shared_pool(saver)
        assert pool is not None and pool.max_size == 3

        opened = []
        real = psycopg_pool.ConnectionPool
        monkeypatch.setattr(psycopg_pool, "ConnectionPool", lambda *a, **k: opened.append(1) or real(*a, **k))
        fetch_checkpoint_threads(uri=postgres_uri, prefix="nobody-", pool=pool)
        assert opened == [], "listing threads must reuse the shared pool"
    finally:
        close_checkpointer(saver)
    assert pool.closed


def test_ui_opens_the_checkpointer_once_across_reruns(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import checkpointer

    calls = []
    real_open = checkpointer.open_checkpointer

    def counting(*args, **kwargs):
        calls.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(checkpointer, "open_checkpointer", counting)
    st.cache_resource.clear()

    app = AppTest.from_file(UI, default_timeout=120).run()
    app.text_input[0].input("alice")
    app.text_input[1].input(TEST_PASSWORD)
    [b for b in app.button if b.label == "Sign in"][0].click().run()
    app.run()
    app.run()
    assert not app.exception
    assert len(calls) == 1, f"every rerun reopened the checkpointer: {len(calls)} opens"
    st.cache_resource.clear()
