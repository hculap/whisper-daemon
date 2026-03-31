"""Entry point for whisper-daemon."""

import argparse
import logging
import queue
import signal
import sys

from whisper_daemon.daemon import Daemon
from whisper_daemon.events import Event
from whisper_daemon.hotkey import HotkeyListener
from whisper_daemon.menubar import run_with_menubar
from whisper_daemon.recorder import AudioRecorder
from whisper_daemon.transcriber import preload_model


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger("whisper_daemon")

    event_queue: queue.Queue[Event] = queue.Queue()

    logger.info("Initializing components...")
    recorder = AudioRecorder(event_queue, silence_timeout=args.silence_timeout)
    hotkey = HotkeyListener(event_queue)
    daemon = Daemon(event_queue, recorder, model=args.model)

    preload_model(args.model)

    def _signal_handler(sig: int, frame: object) -> None:
        logger.info("Received signal %s", signal.Signals(sig).name)
        hotkey.stop()
        daemon.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    hotkey.start()

    print(f"whisper-daemon running — press Cmd+Shift+Space to record")
    print(f"Model: {args.model}")
    print(f"Silence timeout: {args.silence_timeout}s")
    print("Press Ctrl+C to quit")
    print()
    print("NOTE: Terminal needs Accessibility + Microphone permissions")
    print("      System Settings > Privacy & Security > Accessibility")
    print("      System Settings > Privacy & Security > Microphone")

    if args.no_menubar:
        try:
            daemon.run()
        finally:
            hotkey.stop()
    else:
        run_with_menubar(daemon, hotkey)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="whisper-daemon",
        description="Local push-to-talk transcription daemon for macOS",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo-q4",
        help="HuggingFace model repo (default: whisper-large-v3-turbo-q4)",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=1.5,
        help="Seconds of silence before auto-stop (default: 1.5)",
    )
    parser.add_argument(
        "--no-menubar",
        action="store_true",
        help="Run without menu bar icon (CLI-only mode)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("httpcore", "httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
