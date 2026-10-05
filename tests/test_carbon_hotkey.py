"""Tests for Carbon hot key dispatch (the Secure-Input-proof record hotkey)."""

import ctypes

import pytest

from whisper_daemon import carbon_hotkey
from whisper_daemon.carbon_hotkey import CMD_KEY, SHIFT_KEY, CarbonHotkeys


class _FakeCarbon:
    """Stands in for the Carbon dylib; GetEventParameter reports `fired_id`."""

    def __init__(self, register_status: int = 0) -> None:
        self.register_status = register_status
        self.fired_id = 0
        self.fired_signature = carbon_hotkey._SIGNATURE
        self.registered: list[tuple[int, int]] = []
        self.unregistered = 0
        self.handler_removed = False

    def GetApplicationEventTarget(self):
        return 1

    def InstallEventHandler(self, _target, _cb, _n, _spec, _ud, out_ref):
        ctypes.cast(out_ref, ctypes.POINTER(ctypes.c_void_p))[0] = 42
        return 0

    def RemoveEventHandler(self, _ref):
        self.handler_removed = True
        return 0

    def RegisterEventHotKey(self, keycode, modifiers, _hk_id, _target, _opts, _out):
        self.registered.append((keycode, modifiers))
        return self.register_status

    def UnregisterEventHotKey(self, _ref):
        self.unregistered += 1
        return 0

    def GetEventParameter(self, _event, _name, _type, _actual, _size, _out_size, out):
        hk = ctypes.cast(out, ctypes.POINTER(carbon_hotkey._EventHotKeyID)).contents
        hk.signature = self.fired_signature
        hk.id = self.fired_id
        return 0


@pytest.fixture
def fake(monkeypatch):
    lib = _FakeCarbon()
    monkeypatch.setattr(carbon_hotkey, "_load_carbon", lambda: lib)
    return lib


def test_dispatches_to_registered_callback(fake):
    hits: list[str] = []
    hotkeys = CarbonHotkeys()
    hotkeys.register(49, CMD_KEY | SHIFT_KEY, lambda: hits.append("record"))
    hotkeys.register(9, CMD_KEY | SHIFT_KEY, lambda: hits.append("paste"))

    fake.fired_id = 2
    assert hotkeys._on_event(0, 0, 0) == 0

    assert hits == ["paste"]
    assert fake.registered == [(49, 0x300), (9, 0x300)]


def test_ignores_foreign_signature(fake):
    hits: list[str] = []
    hotkeys = CarbonHotkeys()
    hotkeys.register(49, CMD_KEY | SHIFT_KEY, lambda: hits.append("record"))

    fake.fired_id = 1
    fake.fired_signature = 0xDEADBEEF
    hotkeys._on_event(0, 0, 0)

    assert hits == []


def test_callback_exception_does_not_escape(fake):
    def boom() -> None:
        raise RuntimeError("queue closed")

    hotkeys = CarbonHotkeys()
    hotkeys.register(49, CMD_KEY | SHIFT_KEY, boom)
    fake.fired_id = 1

    assert hotkeys._on_event(0, 0, 0) == 0


def test_register_failure_raises(monkeypatch):
    monkeypatch.setattr(carbon_hotkey, "_load_carbon", lambda: _FakeCarbon(register_status=-9878))

    with pytest.raises(OSError, match="RegisterEventHotKey"):
        CarbonHotkeys().register(49, CMD_KEY | SHIFT_KEY, lambda: None)


def test_unregister_all_releases_everything(fake):
    hotkeys = CarbonHotkeys()
    hotkeys.register(49, CMD_KEY | SHIFT_KEY, lambda: None)

    hotkeys.unregister_all()

    assert fake.unregistered == 1
    assert fake.handler_removed is True
    fake.fired_id = 1
    assert hotkeys._callbacks == {}
