"""System-wide hotkeys via Carbon RegisterEventHotKey (ctypes).

Unlike NSEvent global monitors and CGEventTaps, Carbon hot keys keep firing
while another app holds Secure Keyboard Entry (focused password field, iTerm2
auto secure input, a leaked Chrome flag) — verified on macOS 15. The trade-off
is that a registered combo is consumed: the frontmost app no longer sees it.

Callbacks arrive on the main thread via the AppKit run loop, so register only
after NSApplication exists and the run loop is (or will be) running.
"""

import ctypes
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_CARBON_PATH = "/System/Library/Frameworks/Carbon.framework/Carbon"

# Carbon modifier bits (Events.h) — distinct from NSEvent modifier flags.
CMD_KEY = 0x0100
SHIFT_KEY = 0x0200
OPTION_KEY = 0x0800
CONTROL_KEY = 0x1000

_K_EVENT_CLASS_KEYBOARD = int.from_bytes(b"keyb", "big")
_K_EVENT_HOTKEY_PRESSED = 5
_K_EVENT_PARAM_DIRECT_OBJECT = int.from_bytes(b"----", "big")
_TYPE_EVENT_HOTKEY_ID = int.from_bytes(b"hkid", "big")
_SIGNATURE = int.from_bytes(b"WDhk", "big")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)


def _load_carbon() -> ctypes.CDLL:
    lib = ctypes.CDLL(_CARBON_PATH)
    lib.GetApplicationEventTarget.restype = ctypes.c_void_p
    lib.GetApplicationEventTarget.argtypes = []
    lib.InstallEventHandler.restype = ctypes.c_int32
    lib.InstallEventHandler.argtypes = [
        ctypes.c_void_p, _HANDLER, ctypes.c_uint32,
        ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.RemoveEventHandler.restype = ctypes.c_int32
    lib.RemoveEventHandler.argtypes = [ctypes.c_void_p]
    lib.RegisterEventHotKey.restype = ctypes.c_int32
    lib.RegisterEventHotKey.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.UnregisterEventHotKey.restype = ctypes.c_int32
    lib.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
    lib.GetEventParameter.restype = ctypes.c_int32
    lib.GetEventParameter.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return lib


class CarbonHotkeys:
    """Owns a Carbon event handler plus any number of registered hot keys."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        # The ctypes callback must outlive the handler or Carbon calls freed memory.
        self._handler_cb = _HANDLER(self._on_event)
        self._handler_ref = ctypes.c_void_p()
        self._refs: dict[int, ctypes.c_void_p] = {}
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_id = 1

    def _ensure_handler(self) -> ctypes.CDLL:
        if self._lib is not None:
            return self._lib
        lib = _load_carbon()
        spec = _EventTypeSpec(_K_EVENT_CLASS_KEYBOARD, _K_EVENT_HOTKEY_PRESSED)
        status = lib.InstallEventHandler(
            lib.GetApplicationEventTarget(), self._handler_cb, 1,
            ctypes.byref(spec), None, ctypes.byref(self._handler_ref),
        )
        if status != 0:
            raise OSError(f"InstallEventHandler failed: {status}")
        self._lib = lib
        return lib

    def register(self, keycode: int, modifiers: int, callback: Callable[[], None]) -> None:
        """Register a hot key. Raises OSError if Carbon refuses (e.g. combo taken)."""
        lib = self._ensure_handler()
        hotkey_id = self._next_id
        ref = ctypes.c_void_p()
        status = lib.RegisterEventHotKey(
            keycode, modifiers, _EventHotKeyID(_SIGNATURE, hotkey_id),
            lib.GetApplicationEventTarget(), 0, ctypes.byref(ref),
        )
        if status != 0:
            raise OSError(f"RegisterEventHotKey({keycode}, {modifiers:#x}) failed: {status}")
        self._next_id += 1
        self._refs[hotkey_id] = ref
        self._callbacks[hotkey_id] = callback

    def unregister_all(self) -> None:
        if self._lib is None:
            return
        for ref in self._refs.values():
            self._lib.UnregisterEventHotKey(ref)
        self._refs = {}
        self._callbacks = {}
        if self._handler_ref:
            self._lib.RemoveEventHandler(self._handler_ref)
            self._handler_ref = ctypes.c_void_p()
        self._lib = None

    def _on_event(self, _call_ref: int, event: int, _user_data: int) -> int:
        # Never let an exception escape into Carbon — ctypes would only print it.
        try:
            hk = _EventHotKeyID()
            status = self._lib.GetEventParameter(
                event, _K_EVENT_PARAM_DIRECT_OBJECT, _TYPE_EVENT_HOTKEY_ID,
                None, ctypes.sizeof(hk), None, ctypes.byref(hk),
            )
            if status == 0 and hk.signature == _SIGNATURE:
                callback = self._callbacks.get(hk.id)
                if callback is not None:
                    callback()
        except Exception:
            logger.exception("Carbon hotkey callback failed")
        return 0
