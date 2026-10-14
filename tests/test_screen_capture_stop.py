"""Stopping screen capture must neither hang on a stuck capture nor leave a partial PNG."""
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

QUARTZ_NAMES = ('CGDataProviderCopyData', 'CGDisplayBounds', 'CGGetActiveDisplayList',
                'CGImageGetBitsPerPixel', 'CGImageGetBytesPerRow', 'CGImageGetDataProvider',
                'CGImageGetHeight', 'CGImageGetWidth', 'CGWindowListCreateImage',
                'kCGNullWindowID', 'kCGWindowImageDefault', 'kCGWindowListOptionOnScreenOnly')


@pytest.fixture
def screen_capture():
    quartz = sys.modules.get('Quartz') or types.ModuleType('Quartz')  # real one on macOS
    for name in QUARTZ_NAMES:
        if not hasattr(quartz, name):
            setattr(quartz, name, None)
    with patch.dict(sys.modules, {'Quartz': quartz}):
        sys.modules.pop('whisper_daemon.screen_capture', None)
        from whisper_daemon import screen_capture as module
        yield module
    sys.modules.pop('whisper_daemon.screen_capture', None)


def test_stop_is_bounded_and_a_late_capture_writes_nothing(screen_capture, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def hung(display_id):
        entered.set()
        release.wait(5)  # a Quartz call that hangs until long after stop()
        return Image.new('RGB', (32, 32), 'white')

    monkeypatch.setattr(screen_capture, 'STOP_TIMEOUT_S', 0.5)
    monkeypatch.setattr(screen_capture, '_capture_display_quartz', hung)
    monkeypatch.setattr(screen_capture, '_get_display_ids', lambda: [1])
    monkeypatch.setattr(screen_capture, '_get_display_count', lambda: 1)
    capture = screen_capture.ScreenCapture(Path(tmp_path), interval=60)
    capture.start()
    assert entered.wait(2)
    started = time.monotonic()
    capture.stop()
    capture.stop()  # menubar stops it again in its finally block
    assert time.monotonic() - started < 0.9  # one timeout in total, not one per wait
    release.set()
    time.sleep(0.2)
    assert list((tmp_path / 'screenshots').iterdir()) == []  # neither a PNG nor a partial temp


def test_screenshot_appears_atomically(screen_capture, tmp_path, monkeypatch):
    monkeypatch.setattr(screen_capture, '_capture_display_quartz', lambda d: Image.new('RGB', (32, 32), 'white'))
    monkeypatch.setattr(screen_capture, '_get_display_ids', lambda: [1])
    capture = screen_capture.ScreenCapture(Path(tmp_path))
    (tmp_path / 'screenshots').mkdir()
    capture._running = True
    capture._capture_all_displays()
    files = list((tmp_path / 'screenshots').iterdir())
    assert len(files) == 1 and files[0].name.endswith('_display1.png')  # no temp left behind
    assert Image.open(files[0]).size == (32, 32)
