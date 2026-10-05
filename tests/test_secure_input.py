"""Tests for Secure Keyboard Entry detection (the silent hotkey-deafness case)."""

import subprocess
from types import SimpleNamespace

from whisper_daemon import secure_input
from whisper_daemon.secure_input import SecureInputHolder, SecureInputWatch


def test_watch_reports_only_transitions(monkeypatch):
    """poll() is edge-triggered: one 'blocked', silence while held, one 'cleared'."""
    states = iter([False, True, True, True, False, False])
    monkeypatch.setattr(secure_input, "is_enabled", lambda: next(states))
    watch = SecureInputWatch()

    results = [watch.poll() for _ in range(6)]

    assert results == [None, "blocked", None, None, "cleared", None]
    assert watch.blocked is False


def test_watch_blocked_from_first_poll(monkeypatch):
    """A block already active at startup is reported on the very first poll."""
    monkeypatch.setattr(secure_input, "is_enabled", lambda: True)
    watch = SecureInputWatch()

    assert watch.poll() == "blocked"
    assert watch.blocked is True


def _fake_run(outputs):
    def run(cmd, **_kwargs):
        return SimpleNamespace(stdout=outputs[cmd[0]])
    return run


def test_find_holder_resolves_pid_and_app(monkeypatch):
    ioreg_out = '| "IOConsoleUsers" = ({"kCGSSessionSecureInputPID"=105,"kCGSSessionUserIDKey"=501})'
    ps_out = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
    monkeypatch.setattr(subprocess, "run", _fake_run({"ioreg": ioreg_out, "ps": ps_out}))

    assert secure_input.find_holder() == SecureInputHolder(pid=105, app="Google Chrome")


def test_find_holder_none_when_key_absent(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"ioreg": '{"kCGSSessionUserIDKey"=501}'}))

    assert secure_input.find_holder() is None


def test_find_holder_survives_ioreg_failure(monkeypatch):
    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired("ioreg", 10)
    monkeypatch.setattr(subprocess, "run", boom)

    assert secure_input.find_holder() is None


def test_find_holder_flags_exited_process(monkeypatch):
    """A leaked flag whose holder pid is gone is reported as not alive."""
    ioreg_out = '{"kCGSSessionSecureInputPID"=105}'
    monkeypatch.setattr(subprocess, "run", _fake_run({"ioreg": ioreg_out, "ps": ""}))

    holder = secure_input.find_holder()

    assert holder is not None
    assert holder.pid == 105
    assert holder.alive is False
