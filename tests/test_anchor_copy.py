"""The second anchor copy: written with every anchor, checked on its own, and compared with the primary."""
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_lab.config import settings
from rag_lab.security import audit
from rag_lab.security.audit import AnchorCopyError, anchor, prune, record, verify_anchors


@pytest.fixture
def env(tmp_path):
    (tmp_path / "offsite").mkdir()
    return {
        "audit": str(tmp_path / "audit.db"),
        "primary": str(tmp_path / "anchors.jsonl"),
        "copy_dir": str(tmp_path / "offsite"),
        "copy": tmp_path / "offsite" / "anchors.jsonl",
    }


def add_event(env, n=1):
    for i in range(n):
        record("query_received", workspace_id="ops", session_id="s", data={"i": i}, path=env["audit"])


def make_anchor(env):
    return anchor(path=env["audit"], anchor_path=env["primary"], copy_dir=env["copy_dir"])


def verify(env, copy=True):
    return verify_anchors(env["audit"], env["primary"], env["copy_dir"] if copy else None)


def lines(path) -> list[str]:
    return Path(path).read_text().splitlines()


def test_every_anchor_is_written_to_both_files(env):
    add_event(env)
    make_anchor(env)
    add_event(env)
    make_anchor(env)
    assert lines(env["copy"]) == lines(env["primary"]) and len(lines(env["copy"])) == 2
    assert stat.S_IMODE(env["copy"].stat().st_mode) == 0o600
    assert verify(env) == (True, [], 2)


def test_a_copy_that_is_behind_is_reported_and_then_caught_up_by_the_next_anchor(env):
    add_event(env)
    make_anchor(env)
    add_event(env)
    make_anchor(env)
    env["copy"].write_text(lines(env["copy"])[0] + "\n")
    ok, problems, count = verify(env)
    assert not ok and count == 2 and any("behind by 1" in p and "make audit-anchor" in p for p in problems)

    add_event(env)
    make_anchor(env)
    assert lines(env["copy"]) == lines(env["primary"]) and len(lines(env["copy"])) == 3
    assert verify(env)[0]


def test_a_truncated_primary_is_caught_because_the_copy_remembers(env):
    add_event(env)
    make_anchor(env)
    add_event(env)
    make_anchor(env)
    Path(env["primary"]).write_text(lines(env["primary"])[0] + "\n")  # the attacker rewinds
    ok, problems, _ = verify(env)
    assert not ok and any("primary anchor file was truncated" in p for p in problems)
    assert verify(env, copy=False)[0], "without the copy the rewind is invisible: that is the point of it"


def test_a_deleted_primary_is_caught(env):
    add_event(env)
    make_anchor(env)
    Path(env["primary"]).unlink()
    ok, problems, count = verify(env)
    assert not ok and count == 0 and any("primary anchor file was truncated" in p for p in problems)


def test_an_edited_copy_or_primary_is_caught(env):
    add_event(env)
    make_anchor(env)
    original = env["copy"].read_text()
    env["copy"].write_text(original.replace('"seq": 1', '"seq": 9'))
    ok, problems, _ = verify(env)
    assert not ok and any(p.startswith("copy:") and "signature does not match" in p for p in problems)

    env["copy"].write_text(original)
    Path(env["primary"]).write_text(original.replace('"seq": 1', '"seq": 9'))
    ok, problems, _ = verify(env)
    assert not ok and any("signature does not match" in p and not p.startswith("copy:") for p in problems)
    assert any("differs from the primary" in p for p in problems)


def test_a_missing_copy_file_is_reported_once_there_are_anchors(env):
    assert verify(env)[0], "nothing to copy yet"
    add_event(env)
    make_anchor(env)
    env["copy"].unlink()
    ok, problems, _ = verify(env)
    assert not ok and any("does not exist" in p and "1 anchor" in p for p in problems)


def test_an_unreachable_copy_directory_fails_the_command_but_keeps_the_primary(env, tmp_path):
    add_event(env)
    gone = str(tmp_path / "not-mounted" / "anchors")  # neither it nor its parent exists: an unmounted drive
    with pytest.raises(AnchorCopyError, match="not reachable") as caught:
        anchor(path=env["audit"], anchor_path=env["primary"], copy_dir=gone)
    assert len(lines(env["primary"])) == 1 and caught.value.entry["seq"] == 1, "the primary anchor still exists"
    assert not (tmp_path / "not-mounted").exists(), "never create a mount point on the local disk"
    ok, problems, _ = verify_anchors(env["audit"], env["primary"], gone)
    assert not ok and any("not reachable" in p for p in problems)


def test_a_missing_last_directory_is_created_when_its_parent_exists(env, tmp_path):
    add_event(env)
    new_dir = tmp_path / "offsite" / "anchors"
    anchor(path=env["audit"], anchor_path=env["primary"], copy_dir=str(new_dir))
    assert (new_dir / "anchors.jsonl").read_text() == Path(env["primary"]).read_text()


def test_a_copy_that_disagrees_is_never_appended_to(env):
    add_event(env)
    make_anchor(env)
    tampered = env["copy"].read_text().replace('"seq": 1', '"seq": 9')
    env["copy"].write_text(tampered)
    add_event(env)
    with pytest.raises(AnchorCopyError, match="differs from the primary"):
        make_anchor(env)
    assert env["copy"].read_text() == tampered
    assert len(lines(env["primary"])) == 2, "the primary keeps recording"


def test_an_unwritable_copy_is_an_error_not_a_crash(env, monkeypatch):
    add_event(env)
    env["copy"].parent.chmod(0o500)
    try:
        with pytest.raises(AnchorCopyError, match="could not be written"):
            make_anchor(env)
    finally:
        env["copy"].parent.chmod(0o700)


def test_an_explicit_anchor_file_ignores_the_configured_copy_directory(env, tmp_path, monkeypatch):
    """Verifying a backup's anchors must not compare them with the live second copy."""
    monkeypatch.setattr(settings, "audit_anchor_copy_dir", str(tmp_path / "not-mounted" / "x"))
    add_event(env)
    anchor(path=env["audit"], anchor_path=env["primary"])
    assert verify_anchors(env["audit"], env["primary"])[0]
    assert not (tmp_path / "not-mounted").exists()


def test_retention_keeps_both_files_valid(env):
    add_event(env, 3)
    make_anchor(env)
    prune(retention_days=1, path=env["audit"], now=datetime.now(UTC) + timedelta(days=2))
    assert verify(env)[0]
    add_event(env)
    make_anchor(env)
    assert verify(env)[0]


def test_the_command_exits_non_zero_when_the_copy_cannot_be_updated(env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "audit_anchor_copy_dir", str(tmp_path / "not-mounted" / "x"))
    audit.record("query_received", workspace_id="ops", session_id="s", data={"cli": True})
    assert audit.main(["anchor"]) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("Anchored seq ") and "NOT up to date" in captured.err
    assert audit.main(["verify"]) == 1, "verify fails while the copy is unreachable"
    assert "not reachable" in capsys.readouterr().err

    good = tmp_path / "usb"
    good.mkdir()
    monkeypatch.setattr(settings, "audit_anchor_copy_dir", str(good))
    assert audit.main(["anchor"]) == 0
    assert audit.main(["verify"]) == 0
    assert (good / Path(settings.anchor_path).name).exists()
