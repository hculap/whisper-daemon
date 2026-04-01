"""CLI entry point for whisper-daemon."""

import logging
import os
import queue
import signal
import subprocess
import sys
from pathlib import Path

import click

CONFIG_DIR = Path.home() / ".config" / "whisper-daemon"
PID_FILE = CONFIG_DIR / "daemon.pid"
LOG_FILE = CONFIG_DIR / "daemon.log"


@click.group(invoke_without_command=True, context_settings={"show_default": True})
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Local push-to-talk transcription daemon for macOS."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--model", default="mlx-community/whisper-large-v3-turbo-q4", help="HuggingFace model repo.")
@click.option("--silence-timeout", default=0.7, type=float, help="Seconds of silence before auto-stop.")
@click.option("--no-menubar", is_flag=True, help="Run without menu bar icon.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def start(model: str, silence_timeout: float, no_menubar: bool, verbose: bool) -> None:
    """Start the daemon in the background."""
    if _is_running():
        pid = _read_pid()
        click.echo(f"Already running (PID {pid}). Use 'whisper-daemon stop' first.")
        raise SystemExit(1)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "whisper_daemon", "run",
        "--model", model,
        "--silence-timeout", str(silence_timeout),
    ]
    if no_menubar:
        cmd.append("--no-menubar")
    if verbose:
        cmd.append("--verbose")

    log_handle = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
    )

    PID_FILE.write_text(str(proc.pid))
    click.echo(f"Started (PID {proc.pid})")
    click.echo(f"Logs: {LOG_FILE}")
    click.echo(f"Stop: whisper-daemon stop")


@cli.command()
def stop() -> None:
    """Stop the running daemon."""
    if not _is_running():
        click.echo("Not running.")
        return

    pid = _read_pid()
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent SIGTERM to PID {pid}")
    except ProcessLookupError:
        click.echo(f"Process {pid} not found (stale PID file)")

    PID_FILE.unlink(missing_ok=True)


@cli.command()
@click.argument("action", type=click.Choice(["enable", "disable", "status"]))
def autostart(action: str) -> None:
    """Manage auto-start at macOS login.

    \b
    Examples:
      whisper-daemon autostart enable
      whisper-daemon autostart disable
      whisper-daemon autostart status
    """
    from whisper_daemon.autostart import disable, enable, is_enabled

    if action == "enable":
        enable()
        click.echo("Autostart enabled. whisper-daemon will start at login.")
    elif action == "disable":
        disable()
        click.echo("Autostart disabled.")
    elif action == "status":
        if is_enabled():
            click.echo("Autostart: enabled")
        else:
            click.echo("Autostart: disabled")


@cli.command()
def status() -> None:
    """Show daemon status."""
    if _is_running():
        pid = _read_pid()
        click.echo(f"Running (PID {pid})")
    else:
        click.echo("Not running.")


@cli.command()
@click.option("--model", default="mlx-community/whisper-large-v3-turbo-q4", help="HuggingFace model repo.")
@click.option("--silence-timeout", default=0.7, type=float, help="Seconds of silence before auto-stop.")
@click.option("--no-menubar", is_flag=True, help="Run without menu bar icon.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def run(model: str, silence_timeout: float, no_menubar: bool, verbose: bool) -> None:
    """Run the daemon in the foreground."""
    _setup_logging(verbose)
    logger = logging.getLogger("whisper_daemon")

    from whisper_daemon.daemon import Daemon
    from whisper_daemon.events import Event
    from whisper_daemon.hotkey import HotkeyListener
    from whisper_daemon.menubar import run_with_menubar
    from whisper_daemon.config import load_settings
    from whisper_daemon.recorder import AudioRecorder
    from whisper_daemon.transcriber import preload_model

    settings = load_settings()
    device = settings.recording_device or None

    event_queue: queue.Queue[Event] = queue.Queue()

    logger.info("Initializing components...")
    recorder = AudioRecorder(event_queue, silence_timeout=silence_timeout, device=device)
    hotkey = HotkeyListener(event_queue)
    daemon = Daemon(event_queue, recorder, model=model)

    preload_model(model)

    screen_capture = None
    activity_monitor = None

    def _signal_handler(sig: int, frame: object) -> None:
        logger.info("Received signal %s", signal.Signals(sig).name)
        hotkey.stop()
        if activity_monitor:
            activity_monitor.stop()
        if screen_capture:
            screen_capture.stop()
        daemon.shutdown()
        if not no_menubar:
            from PyObjCTools import AppHelper
            AppHelper.stopEventLoop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if settings.capture_screenshots:
        from whisper_daemon.screen_capture import ScreenCapture

        screen_capture = ScreenCapture(
            output_dir=settings.recording_dir_path,
            interval=settings.screenshot_interval,
        )
        screen_capture.start()

        if settings.screenshot_event_triggers:
            from whisper_daemon.activity_monitor import ActivityMonitor

            activity_monitor = ActivityMonitor(
                on_activity=screen_capture.capture_now,
                debounce_delay=settings.screenshot_debounce,
                cooldown=settings.screenshot_cooldown,
            )

    hotkey.start()

    click.echo(f"whisper-daemon running — press Cmd+Shift+Space to record")
    click.echo(f"Model: {model}")
    click.echo(f"Silence timeout: {silence_timeout}s")
    if screen_capture:
        trigger_mode = "events + interval" if activity_monitor else "interval only"
        click.echo(f"Screenshots: {trigger_mode}")
    click.echo("Press Ctrl+C to quit")

    if no_menubar:
        if activity_monitor:
            activity_monitor.start()
        try:
            daemon.run()
        finally:
            if activity_monitor:
                activity_monitor.stop()
            if screen_capture:
                screen_capture.stop()
            hotkey.stop()
    else:
        on_ready = activity_monitor.start if activity_monitor else None
        run_with_menubar(daemon, hotkey, on_appkit_ready=on_ready)


@cli.command()
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f).")
@click.option("-n", "--lines", default=50, type=int, help="Number of lines to show.")
def logs(follow: bool, lines: int) -> None:
    """Show daemon logs."""
    if not LOG_FILE.exists():
        click.echo("No log file found. Start the daemon first.")
        return

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(LOG_FILE))

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


@cli.command()
def devices() -> None:
    """List available audio input devices."""
    import sounddevice as sd

    devs = sd.query_devices()
    click.echo("Available input devices:\n")
    click.echo(f"  {'ID':<4} {'Name':<45} {'Channels':<10} {'Sample Rate'}")
    click.echo(f"  {'--':<4} {'----':<45} {'--------':<10} {'-----------'}")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            default = " *" if i == sd.default.device[0] else ""
            click.echo(
                f"  {i:<4} {d['name']:<45} {d['max_input_channels']:<10} {d['default_samplerate']:.0f} Hz{default}"
            )
    click.echo("\n  * = system default")
    click.echo("\n  Tip: For system audio capture, install BlackHole:")
    click.echo("    brew install blackhole-2ch")


AUDIO_VIDEO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm",
    ".mp4", ".mkv", ".avi", ".mov", ".aac", ".wma",
}


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "formats", default="txt", help="Output formats, comma-separated: txt,srt,vtt,json.")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory (default: same as input).")
@click.option("--model", default="mlx-community/whisper-large-v3-turbo-q4", help="HuggingFace model repo.")
@click.option("--language", default=None, help="Force language code (e.g. pl, en). Default: auto-detect.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def transcribe(
    paths: tuple[str, ...],
    formats: str,
    output_dir: str | None,
    model: str,
    language: str | None,
    verbose: bool,
) -> None:
    """Transcribe audio/video files to text, SRT, VTT, or JSON.

    Accepts files or directories. Directories are scanned for audio/video files.
    Requires ffmpeg installed on the system.

    \b
    Examples:
      whisper-daemon transcribe meeting.mp3
      whisper-daemon transcribe meeting.mp3 --format srt
      whisper-daemon transcribe meeting.mp3 --format txt,srt,json
      whisper-daemon transcribe ./recordings/
      whisper-daemon transcribe video.mp4 --output-dir ./out
    """
    _setup_logging(verbose)

    from whisper_daemon.formats import FORMATTERS
    from whisper_daemon.transcriber import preload_model, transcribe_file

    format_list = [f.strip().lower() for f in formats.split(",")]
    for fmt in format_list:
        if fmt not in FORMATTERS:
            click.echo(f"Unknown format: {fmt}. Available: {', '.join(FORMATTERS)}")
            raise SystemExit(1)

    files = _collect_files(paths)
    if not files:
        click.echo("No audio/video files found.")
        raise SystemExit(1)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Model: {model}")
    click.echo(f"Files: {len(files)}")
    click.echo(f"Formats: {', '.join(format_list)}")
    click.echo()

    preload_model(model)

    for i, file_path in enumerate(files, start=1):
        click.echo(f"[{i}/{len(files)}] {file_path.name}")
        try:
            result = transcribe_file(str(file_path), model=model, language=language)
        except Exception as exc:
            click.echo(f"  ERROR: {exc}")
            continue

        dest_dir = out_dir or file_path.parent
        stem = file_path.stem

        for fmt in format_list:
            output_text = FORMATTERS[fmt](result)
            output_path = dest_dir / f"{stem}.{fmt}"
            output_path.write_text(output_text, encoding="utf-8")
            click.echo(f"  → {output_path}")

    click.echo(f"\nDone — {len(files)} file(s) transcribed.")


@cli.command()
@click.argument("output", default="recording.txt", type=click.Path())
@click.option("--device", default=None, help="Audio device name or index (default: system default). Use 'devices' to list.")
@click.option("--format", "formats", default="txt", help="Output formats, comma-separated: txt,srt,vtt,json.")
@click.option("--model", default="mlx-community/whisper-large-v3-turbo-q4", help="HuggingFace model repo.")
@click.option("--language", default=None, help="Force language code (e.g. pl, en). Default: auto-detect.")
@click.option("--chunk-silence", default=1.0, type=float, help="Seconds of silence to split chunks.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def record(
    output: str,
    device: str | None,
    formats: str,
    model: str,
    language: str | None,
    chunk_silence: float,
    verbose: bool,
) -> None:
    """Record and transcribe a live meeting or long audio session.

    Records continuously until Ctrl+C. Splits audio at natural pauses
    and transcribes chunks in parallel for fast results.

    \b
    Examples:
      whisper-daemon record                                   # → recording.txt
      whisper-daemon record meeting.txt --format txt,srt
      whisper-daemon record meeting.txt --device "BlackHole 2ch"
      whisper-daemon record meeting.txt --device 3            # by device index
    """
    import concurrent.futures
    import queue as queue_mod

    from whisper_daemon.formats import FORMATTERS
    from whisper_daemon.meeting_recorder import AudioChunk, MeetingRecorder
    from whisper_daemon.transcriber import preload_model, transcribe as transcribe_audio

    _setup_logging(verbose)

    format_list = [f.strip().lower() for f in formats.split(",")]
    for fmt in format_list:
        if fmt not in FORMATTERS:
            click.echo(f"Unknown format: {fmt}. Available: {', '.join(FORMATTERS)}")
            raise SystemExit(1)

    # Resolve device (allow numeric index)
    resolved_device: str | int | None = device
    if device is not None:
        try:
            resolved_device = int(device)
        except ValueError:
            resolved_device = device

    output_path = Path(output)
    output_stem = output_path.stem
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Model: {model}")
    click.echo(f"Device: {device or 'system default'}")
    click.echo(f"Chunk silence: {chunk_silence}s")
    click.echo(f"Output: {output_path}")
    click.echo()

    preload_model(model)

    chunk_queue: queue_mod.Queue[AudioChunk | None] = queue_mod.Queue()
    recorder = MeetingRecorder(chunk_queue, device=resolved_device, chunk_silence=chunk_silence)

    all_results: list[tuple[float, dict]] = []  # (start_time, result)
    chunk_count = 0

    click.echo("Recording... press Ctrl+C to stop.\n")
    recorder.start()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures: dict[concurrent.futures.Future, float] = {}

    try:
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.5)
            except queue_mod.Empty:
                _collect_results(futures, all_results)
                continue

            if chunk is None:
                break

            chunk_count += 1
            click.echo(f"  Chunk {chunk_count}: {chunk.duration:.1f}s (at {chunk.start_time:.1f}s)")

            future = pool.submit(transcribe_audio, chunk.audio, model)
            futures[future] = chunk.start_time

            _collect_results(futures, all_results)

    except KeyboardInterrupt:
        click.echo("\n\nStopping recording...")
        recorder.stop()

        # Wait for in-flight transcription FIRST (avoid concurrent GPU access)
        _collect_results(futures, all_results, wait=True)

        # Now safe to transcribe remaining chunks
        while True:
            try:
                chunk = chunk_queue.get_nowait()
            except queue_mod.Empty:
                break
            if chunk is None:
                break
            chunk_count += 1
            click.echo(f"  Chunk {chunk_count}: {chunk.duration:.1f}s (at {chunk.start_time:.1f}s)")
            result_text = transcribe_audio(chunk.audio, model)
            if result_text:
                all_results.append((chunk.start_time, {"text": result_text, "segments": [], "language": ""}))
    finally:
        pool.shutdown(wait=False)

    if not all_results:
        click.echo("No audio transcribed.")
        return

    merged = _merge_results(all_results)

    for fmt in format_list:
        ext_path = output_dir / f"{output_stem}.{fmt}"
        content = FORMATTERS[fmt](merged)
        ext_path.write_text(content, encoding="utf-8")
        click.echo(f"  → {ext_path}")

    total_duration = sum(r[0] for r in all_results) if not all_results else max(r[0] for r in all_results)
    click.echo(f"\nDone — {chunk_count} chunks, {len(merged.get('segments', []))} segments.")


def _collect_results(
    futures: dict,
    all_results: list[tuple[float, dict]],
    wait: bool = False,
) -> None:
    """Collect completed transcription futures."""
    if wait and futures:
        concurrent_futures = __import__("concurrent.futures", fromlist=["concurrent"])
        done, _ = concurrent_futures.wait(futures.keys())
    else:
        done = {f for f in futures if f.done()}

    for future in done:
        start_time = futures.pop(future)
        try:
            text = future.result()
            if text:
                all_results.append((start_time, {"text": text, "segments": [], "language": ""}))
        except Exception as exc:
            logging.getLogger(__name__).error("Chunk transcription failed: %s", exc)


def _merge_results(results: list[tuple[float, dict]]) -> dict:
    """Merge chunk results into a single result dict with adjusted timestamps."""
    from whisper_daemon.formats import merge_chunk_results
    return merge_chunk_results(results)


def _collect_files(paths: tuple[str, ...]) -> list[Path]:
    """Resolve paths to a flat list of audio/video files."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in AUDIO_VIDEO_EXTENSIONS:
                    files.append(child)
    return files


def _read_pid() -> int:
    return int(PID_FILE.read_text().strip())


def _is_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = _read_pid()
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False


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
    cli()
