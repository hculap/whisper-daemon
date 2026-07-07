"""Global PortAudio lifecycle management with CoreAudio topology tracking.

PortAudio snapshots the CoreAudio device list at Pa_Initialize. After a
device change (Bluetooth headset connect/disconnect, default input switch)
that snapshot is stale: opening "system default" can succeed yet bind a
dead device that delivers pure silence. The only reliable refresh is a
full ``Pa_Terminate``/``Pa_Initialize`` cycle — which invalidates every
live stream, so it must never run while another recording is active.

This module owns both concerns:

- A CoreAudio property listener marks the topology "dirty" the moment the
  device list or the default input device changes.
- A registry of live input streams gates the refresh: ``ensure_fresh()``
  reinitializes PortAudio only when the topology is dirty AND no stream
  is registered.

Recorders must call ``ensure_fresh()`` before opening a stream and
``register_stream()``/``unregister_stream()`` around its lifetime.
"""

import contextlib
import ctypes
import logging
import threading
import time

import sounddevice as sd

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Serializes a Pa_Terminate/Pa_Initialize cycle against ENTIRE stream-open
# sequences. Registration happens only after sd.InputStream()+start(), so
# without this a concurrent ensure_fresh() on another thread could tear
# PortAudio down under a stream that is open but not yet registered —
# undefined behavior / segfault. Reentrant because the open paths hold it
# while _open_stream() re-enters ensure_fresh() on their retry branches.
# Lock order everywhere: _open_lock -> _lock.
_open_lock = threading.RLock()
_live_streams: set[int] = set()
_topology_dirty = False
_last_refresh: float = 0.0
_listener_installed = False


@contextlib.contextmanager
def stream_open():
    """Hold the open-lock across an open→start→register sequence.

    Recorders wrap their whole open sequence in this so a PortAudio
    refresh can never run while a stream of theirs is half-open.
    """
    with _open_lock:
        yield

# Strong references so ctypes callbacks/structs outlive registration.
_listener_refs: list[object] = []


def _fourcc(code: str) -> int:
    """Convert a 4-char CoreAudio code like 'dIn ' to its UInt32 value."""
    return int.from_bytes(code.encode("ascii"), "big")


_kAudioObjectSystemObject = 1
_kAudioObjectPropertyScopeGlobal = _fourcc("glob")
_kAudioObjectPropertyElementMain = 0
_kAudioHardwarePropertyDevices = _fourcc("dev#")
_kAudioHardwarePropertyDefaultInputDevice = _fourcc("dIn ")


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


_LISTENER_PROC = ctypes.CFUNCTYPE(
    ctypes.c_int32,  # OSStatus
    ctypes.c_uint32,  # AudioObjectID
    ctypes.c_uint32,  # inNumberAddresses
    ctypes.POINTER(_AudioObjectPropertyAddress),
    ctypes.c_void_p,
)


def install_topology_listener() -> bool:
    """Register CoreAudio listeners for device-list and default-input changes.

    Returns True on success. On failure the module still works: callers
    can mark the topology dirty manually via ``mark_dirty()`` whenever a
    stream open fails or a recording turns out silent.
    """
    global _listener_installed
    if _listener_installed:
        return True

    try:
        coreaudio = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        add_listener = coreaudio.AudioObjectAddPropertyListener
        add_listener.restype = ctypes.c_int32
        add_listener.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_AudioObjectPropertyAddress),
            _LISTENER_PROC,
            ctypes.c_void_p,
        ]

        @_LISTENER_PROC
        def _on_change(object_id, num_addresses, addresses, client_data):
            mark_dirty("CoreAudio topology change")
            return 0

        # Keep the callback alive BEFORE registering it: if a later
        # registration in the loop fails and we return, the first listener
        # is already live in CoreAudio and must not point at a GC'd callback
        # (that would be a use-after-free on the next device change).
        _listener_refs.append(_on_change)

        for selector in (
            _kAudioHardwarePropertyDevices,
            _kAudioHardwarePropertyDefaultInputDevice,
        ):
            address = _AudioObjectPropertyAddress(
                selector,
                _kAudioObjectPropertyScopeGlobal,
                _kAudioObjectPropertyElementMain,
            )
            status = add_listener(
                _kAudioObjectSystemObject, ctypes.byref(address), _on_change, None
            )
            if status != 0:
                logger.warning(
                    "CoreAudio listener registration failed (selector=%#x, status=%d)",
                    selector, status,
                )
                return False
            _listener_refs.append(address)

        _listener_installed = True
        logger.info("CoreAudio device-topology listener installed")
        return True
    except Exception:
        logger.exception("Could not install CoreAudio topology listener")
        return False


def mark_dirty(reason: str) -> None:
    """Flag the PortAudio device snapshot as stale."""
    global _topology_dirty
    if not _topology_dirty:
        logger.info("Audio topology marked dirty: %s", reason)
    _topology_dirty = True


def register_stream(stream: object) -> None:
    """Track a live input stream; blocks PortAudio refresh while present."""
    with _lock:
        _live_streams.add(id(stream))


def unregister_stream(stream: object) -> None:
    """Remove a stream from the live registry."""
    with _lock:
        _live_streams.discard(id(stream))


def live_stream_count() -> int:
    with _lock:
        return len(_live_streams)


def ensure_fresh(force: bool = False) -> bool:
    """Reinitialize PortAudio if the device topology changed since last init.

    Safe to call before every stream open — it is a no-op unless the
    topology is dirty (or ``force``) and refuses to run while any
    registered stream is live. Returns True if a refresh happened.
    """
    global _topology_dirty, _last_refresh
    # Acquire the open-lock FIRST (lock order _open_lock -> _lock) so no
    # recorder can be mid-open when we terminate PortAudio. Recorders hold
    # _open_lock across their whole open→start→register sequence, so once
    # we hold it every stream is either fully registered (and blocks us
    # via _live_streams) or not yet created.
    with _open_lock:
        with _lock:
            if not (_topology_dirty or force):
                return False
            if _live_streams:
                logger.warning(
                    "PortAudio refresh needed but %d stream(s) live — deferring",
                    len(_live_streams),
                )
                return False
            try:
                # Clear BEFORE reinit: a topology change landing during the
                # refresh must survive and trigger a follow-up, not be lost.
                _topology_dirty = False
                sd._terminate()
                sd._initialize()
                _last_refresh = time.monotonic()
                logger.info("PortAudio reinitialized — device list refreshed")
                return True
            except Exception:
                _topology_dirty = True
                logger.exception("PortAudio reinitialize failed")
                return False
