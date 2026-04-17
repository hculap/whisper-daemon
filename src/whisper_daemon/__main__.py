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
NATIVE_LOG_FILE = CONFIG_DIR / "daemon.native.log"
NATIVE_LOG_MAX_BYTES = 10 * 1024 * 1024


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

    # Capture native stdout/stderr (PortAudio, Metal, crash messages) to a
    # dedicated file — Python logging goes to daemon.log via a handler, but
    # anything the C libraries write directly to fd 1/2 bypasses that. If we
    # dropped it to /dev/null (as this code previously did) a real crash would
    # leave zero evidence. Simple size-based rotation: on each start, move the
    # existing file aside if it has grown past the cap.
    if NATIVE_LOG_FILE.exists() and NATIVE_LOG_FILE.stat().st_size > NATIVE_LOG_MAX_BYTES:
        NATIVE_LOG_FILE.replace(CONFIG_DIR / "daemon.native.log.old")
    native_log = open(NATIVE_LOG_FILE, "a", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=native_log,
        stderr=native_log,
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
    daemon = Daemon(event_queue, recorder, model=model, settings=settings)

    preload_model(model)

    menubar_delegate = None

    def _signal_handler(sig: int, frame: object) -> None:
        logger.info("Received signal %s", signal.Signals(sig).name)
        hotkey.stop()
        if menubar_delegate is not None:
            menubar_delegate.graceful_stop()
        daemon.shutdown()
        if not no_menubar:
            from PyObjCTools import AppHelper
            AppHelper.stopEventLoop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    click.echo(f"whisper-daemon running — press Cmd+Shift+Space to record")
    click.echo(f"Model: {model}")
    click.echo(f"Silence timeout: {silence_timeout}s")
    click.echo("Press Ctrl+C to quit")

    if no_menubar:
        hotkey.start()
        try:
            daemon.run()
        finally:
            hotkey.stop()
    else:
        def on_appkit_ready() -> None:
            hotkey.start()

        def on_delegate_ready(delegate: object) -> None:
            nonlocal menubar_delegate
            menubar_delegate = delegate

        run_with_menubar(
            daemon, hotkey,
            on_appkit_ready=on_appkit_ready,
            on_delegate_ready=on_delegate_ready,
        )


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
@click.option("--diarize", is_flag=True, help="Enable speaker diarization (requires HF_TOKEN).")
@click.option("--num-speakers", type=int, default=None, help="Expected number of speakers (improves diarization).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def transcribe(
    paths: tuple[str, ...],
    formats: str,
    output_dir: str | None,
    model: str,
    language: str | None,
    diarize: bool,
    num_speakers: int | None,
    verbose: bool,
) -> None:
    """Transcribe audio/video files to text, SRT, VTT, or JSON.

    Accepts files or directories. Directories are scanned for audio/video files.
    Requires ffmpeg installed on the system.

    \b
    Examples:
      whisper-daemon transcribe meeting.mp3
      whisper-daemon transcribe meeting.mp3 --format srt
      whisper-daemon transcribe meeting.mp3 --diarize --format txt,json
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
    if diarize:
        click.echo("Diarization: enabled")
    click.echo()

    preload_model(model)

    for i, file_path in enumerate(files, start=1):
        click.echo(f"[{i}/{len(files)}] {file_path.name}")
        try:
            result = transcribe_file(str(file_path), model=model, language=language)
        except Exception as exc:
            click.echo(f"  ERROR: {exc}")
            continue

        if diarize:
            from whisper_daemon.diarizer import diarize_file as run_diarization
            from whisper_daemon.diarize_merge import merge_speakers_with_transcript

            click.echo("  Diarizing speakers...")
            speaker_segments = run_diarization(str(file_path), num_speakers=num_speakers)
            result = merge_speakers_with_transcript(speaker_segments, result)
            speaker_count = len(result.get("speakers", []))
            click.echo(f"  Found {speaker_count} speaker(s)")

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
@click.option("--diarize", is_flag=True, help="Enable speaker diarization (requires HF_TOKEN).")
@click.option("--diarize-mode", type=click.Choice(["batch", "realtime", "hybrid"]), default="hybrid", help="Diarization approach.")
@click.option("--num-speakers", type=int, default=None, help="Expected number of speakers (improves diarization).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def record(
    output: str,
    device: str | None,
    formats: str,
    model: str,
    language: str | None,
    chunk_silence: float,
    diarize: bool,
    diarize_mode: str,
    num_speakers: int | None,
    verbose: bool,
) -> None:
    """Record and transcribe a live meeting or long audio session.

    Records continuously until Ctrl+C. Splits audio at natural pauses
    and transcribes chunks in parallel for fast results.

    \b
    Examples:
      whisper-daemon record                                   # → recording.txt
      whisper-daemon record meeting.txt --format txt,srt
      whisper-daemon record meeting.txt --diarize             # hybrid (default)
      whisper-daemon record meeting.txt --diarize --diarize-mode realtime
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
    if diarize:
        click.echo(f"Diarization: {diarize_mode}")
    click.echo()

    preload_model(model)

    # Set up realtime speaker tracker if needed
    speaker_tracker = None
    if diarize and diarize_mode in ("realtime", "hybrid"):
        from whisper_daemon.diarizer import SpeakerTracker
        speaker_tracker = SpeakerTracker()

    chunk_queue: queue_mod.Queue[AudioChunk | None] = queue_mod.Queue()
    recorder = MeetingRecorder(chunk_queue, device=resolved_device, chunk_silence=chunk_silence)

    all_results: list[tuple[float, dict]] = []  # (start_time, result)
    all_audio_chunks: list[tuple[float, np.ndarray]] = []  # for batch diarization
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

            if diarize:
                all_audio_chunks.append((chunk.start_time, chunk.audio.copy()))

            if speaker_tracker is not None:
                rt_segments = speaker_tracker.identify(chunk.audio, chunk.start_time)
                if rt_segments:
                    click.echo(f"    → Speaker {rt_segments[0].speaker + 1}")

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
            if diarize:
                all_audio_chunks.append((chunk.start_time, chunk.audio.copy()))
            if speaker_tracker is not None:
                speaker_tracker.identify(chunk.audio, chunk.start_time)
            result_text = transcribe_audio(chunk.audio, model)
            if result_text:
                all_results.append((chunk.start_time, {"text": result_text, "segments": [], "language": ""}))
    finally:
        pool.shutdown(wait=False)

    if not all_results:
        click.echo("No audio transcribed.")
        return

    merged = _merge_results(all_results)

    if diarize:
        from whisper_daemon.diarize_merge import merge_speakers_with_transcript

        if diarize_mode in ("batch", "hybrid"):
            from whisper_daemon.diarizer import diarize_batch

            full_audio = _concatenate_audio_chunks(all_audio_chunks)
            click.echo(f"\nDiarizing {len(full_audio) / 16000:.1f}s of audio (batch correction)...")
            batch_segments = diarize_batch(full_audio, num_speakers=num_speakers)
            final_result = merge_speakers_with_transcript(batch_segments, merged)
            speaker_count = len(final_result.get("speakers", []))
            click.echo(f"Found {speaker_count} speaker(s)")

            for fmt in format_list:
                ext_path = output_dir / f"{output_stem}.{fmt}"
                content = FORMATTERS[fmt](final_result)
                ext_path.write_text(content, encoding="utf-8")
                click.echo(f"  → {ext_path}")

        elif diarize_mode == "realtime" and speaker_tracker is not None:
            rt_segments = speaker_tracker.get_all_segments()
            rt_result = merge_speakers_with_transcript(rt_segments, merged)
            speaker_count = len(rt_result.get("speakers", []))
            click.echo(f"\nFound {speaker_count} speaker(s) (realtime)")

            for fmt in format_list:
                ext_path = output_dir / f"{output_stem}.{fmt}"
                content = FORMATTERS[fmt](rt_result)
                ext_path.write_text(content, encoding="utf-8")
                click.echo(f"  → {ext_path}")
    else:
        for fmt in format_list:
            ext_path = output_dir / f"{output_stem}.{fmt}"
            content = FORMATTERS[fmt](merged)
            ext_path.write_text(content, encoding="utf-8")
            click.echo(f"  → {ext_path}")

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


def _concatenate_audio_chunks(chunks: list[tuple[float, "np.ndarray"]]) -> "np.ndarray":
    """Concatenate audio chunks into a single continuous array.

    Chunks may overlap; places each chunk at its start_time offset
    to reconstruct the original timeline.
    """
    import numpy as np

    if not chunks:
        return np.array([], dtype=np.float32)

    sorted_chunks = sorted(chunks, key=lambda c: c[0])
    last_start, last_audio = sorted_chunks[-1]
    total_samples = int((last_start + len(last_audio) / 16000) * 16000) + 1
    full_audio = np.zeros(total_samples, dtype=np.float32)

    for start_time, audio in sorted_chunks:
        offset = int(start_time * 16000)
        end = offset + len(audio)
        if end > len(full_audio):
            full_audio = np.pad(full_audio, (0, end - len(full_audio)))
        full_audio[offset:end] = audio

    return full_audio


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
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    # Clear any pre-existing handlers to avoid duplicates
    root.handlers.clear()
    root.setLevel(level)

    # Console handler (stderr) — visible in foreground mode
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (canonical log location)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("httpcore", "httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker:.*leaked semaphore", category=UserWarning)


if __name__ == "__main__":
    cli()
