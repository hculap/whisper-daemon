from types import SimpleNamespace

import pytest


GIB = 1024**3


def test_high_footprint_reclaims_then_restarts_when_idle():
    from whisper_daemon.memory_guard import MemoryGuard

    actions = []
    guard = MemoryGuard(
        measure=lambda: 5 * GIB,
        is_idle=lambda: True,
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: actions.append(("restart", reason)),
        limit_bytes=4 * GIB,
        startup_grace_sec=0,
    )

    guard.check()
    assert actions == ["reclaim"]

    guard.check()
    assert actions[1][0] == "restart"
    assert "5.0 GiB" in actions[1][1]
    assert "4.0 GiB" in actions[1][1]


def test_high_footprint_is_deferred_until_application_is_idle():
    from whisper_daemon.memory_guard import MemoryGuard

    state = {"idle": False}
    actions = []
    guard = MemoryGuard(
        measure=lambda: 5 * GIB,
        is_idle=lambda: state["idle"],
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: actions.append("restart"),
        limit_bytes=4 * GIB,
        startup_grace_sec=0,
    )

    guard.check()
    guard.check()
    assert actions == []

    state["idle"] = True
    guard.check()
    assert actions == ["reclaim"]
    guard.check()
    assert actions == ["reclaim", "restart"]


def test_memory_recovery_cancels_a_pending_restart():
    from whisper_daemon.memory_guard import MemoryGuard

    readings = iter([5 * GIB, 3 * GIB, 5 * GIB])
    actions = []
    guard = MemoryGuard(
        measure=lambda: next(readings),
        is_idle=lambda: True,
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: actions.append("restart"),
        limit_bytes=4 * GIB,
        startup_grace_sec=0,
    )

    guard.check()
    guard.check()
    guard.check()
    assert actions == ["reclaim", "reclaim"]


def test_deferred_reclaim_is_retried_before_restart():
    from whisper_daemon.memory_guard import MemoryGuard

    outcomes = iter([False, True])
    actions = []

    def reclaim():
        actions.append("reclaim")
        return next(outcomes)

    guard = MemoryGuard(
        measure=lambda: 5 * GIB,
        is_idle=lambda: True,
        reclaim=reclaim,
        restart=lambda reason: actions.append("restart"),
        limit_bytes=4 * GIB,
        startup_grace_sec=0,
    )

    guard.check()
    guard.check()
    assert actions == ["reclaim", "reclaim"]
    guard.check()
    assert actions == ["reclaim", "reclaim", "restart"]


def test_reclaim_safety_gate_never_blocks_the_watchdog():
    import threading
    from whisper_daemon import supervisor

    activity_entered = threading.Event()
    allow_activity = threading.Event()
    actions = []

    def hold_activity_gate():
        with supervisor.activity_transition():
            activity_entered.set()
            assert allow_activity.wait(2)

    activity = threading.Thread(target=hold_activity_gate)
    activity.start()
    assert activity_entered.wait(2)

    watchdog = threading.Thread(
        target=lambda: actions.append(
            supervisor.reclaim_if_safe(
                lambda: actions.append("reclaim"),
                lambda: True,
            )
        )
    )
    watchdog.start()
    try:
        watchdog.join(0.2)
        assert not watchdog.is_alive()
        assert actions == [False]
    finally:
        allow_activity.set()
        activity.join(2)

    assert supervisor.reclaim_if_safe(
        lambda: actions.append("reclaim"),
        lambda: True,
    )
    assert actions == [False, "reclaim"]


@pytest.mark.parametrize("measurement", [None, RuntimeError("unavailable")])
def test_unavailable_footprint_never_restarts(measurement):
    from whisper_daemon.memory_guard import MemoryGuard

    actions = []

    def measure():
        if isinstance(measurement, Exception):
            raise measurement
        return measurement

    guard = MemoryGuard(
        measure=measure,
        is_idle=lambda: True,
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: actions.append("restart"),
        startup_grace_sec=0,
    )
    guard.check()
    guard.check()
    assert actions == []


def test_startup_grace_prevents_a_preload_restart():
    from whisper_daemon.memory_guard import MemoryGuard

    now = {"value": 100.0}
    actions = []
    guard = MemoryGuard(
        measure=lambda: 5 * GIB,
        is_idle=lambda: True,
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: actions.append("restart"),
        limit_bytes=4 * GIB,
        startup_grace_sec=300,
        clock=lambda: now["value"],
    )

    now["value"] = 399.0
    guard.check()
    assert actions == []
    now["value"] = 400.0
    guard.check()
    assert actions == ["reclaim"]


def test_restart_safety_requires_all_user_work_to_be_idle():
    import ast
    import threading
    from pathlib import Path
    from whisper_daemon.daemon import State

    tree = ast.parse(Path("src/whisper_daemon/menubar.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MenuBarDelegate")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "memory_restart_safe")
    scope = {"State": State}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "menubar.py", "exec"), scope)
    restart_safe = scope["memory_restart_safe"]

    obj = SimpleNamespace(
        _daemon=SimpleNamespace(_state=State.IDLE),
        _meeting_phase="idle",
        _meeting_thread=None,
        _browser_start_pending=None,
        _browser_prebuffer_bytes=0,
        _file_jobs=0,
        _file_jobs_lock=threading.Lock(),
        _settings_jobs=0,
        _settings_lock=threading.Lock(),
    )
    assert restart_safe(obj)

    for field, value in (
        ("daemon", State.REVIEW),
        ("meeting_phase", "transcribing"),
        ("meeting_thread", SimpleNamespace(is_alive=lambda: True)),
        ("browser_start_pending", "session"),
        ("browser_prebuffer_bytes", 4),
        ("file_jobs", 1),
        ("settings_jobs", 1),
    ):
        changed = SimpleNamespace(**obj.__dict__)
        if field == "daemon":
            changed._daemon = SimpleNamespace(_state=value)
        else:
            setattr(changed, "_" + field, value)
        assert not restart_safe(changed), field


def test_extension_setting_is_busy_before_worker_start_and_clears_after_persist():
    import ast
    import logging
    import threading
    from pathlib import Path
    from types import MethodType
    from whisper_daemon import supervisor
    from whisper_daemon.bridge_protocol import apply_settings_patch
    from whisper_daemon.config import Settings
    from whisper_daemon.daemon import State

    tree = ast.parse(Path("src/whisper_daemon/menubar.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MenuBarDelegate")
    wanted = {"_persist_and_push_settings", "_on_browser_control", "memory_restart_safe"}
    functions = {
        n.name: n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    }
    saved = []
    scheduled = []
    scope = {
        "AppHelper": SimpleNamespace(callAfter=lambda *args: None),
        "State": State,
        "apply_settings_patch": apply_settings_patch,
        "logger": logging.getLogger(__name__),
        "save_settings": lambda settings: saved.append(settings.capture_mic),
        "supervisor": supervisor,
    }
    for function in functions.values():
        exec(compile(ast.Module(body=[function], type_ignores=[]), "menubar.py", "exec"), scope)

    settings = Settings()
    obj = SimpleNamespace(
        _settings=settings,
        _settings_lock=threading.Lock(),
        _settings_jobs=0,
        _daemon=SimpleNamespace(_state=State.IDLE, _settings=settings),
        _meeting_phase="idle",
        _meeting_thread=None,
        _browser_start_pending=None,
        _browser_prebuffer_bytes=0,
        _file_jobs=0,
        _file_jobs_lock=threading.Lock(),
        _run_off_bridge=lambda fn, *args: scheduled.append((fn, args)),
        _push_settings_snapshot=lambda: None,
        _sync_menu_from_settings=lambda: None,
    )
    obj._persist_and_push_settings = MethodType(scope["_persist_and_push_settings"], obj)
    obj.memory_restart_safe = MethodType(scope["memory_restart_safe"], obj)

    scope["_on_browser_control"](obj, {"type": "set_settings", "patch": {"capture_mic": False}})

    assert obj._settings_jobs == 1
    assert not obj.memory_restart_safe()
    assert len(scheduled) == 1
    scheduled[0][0](*scheduled[0][1])
    assert saved == [False]
    assert obj._settings_jobs == 0
    assert obj.memory_restart_safe()


def test_native_setting_mutation_and_persistence_share_restart_gate():
    import ast
    import threading
    from collections.abc import Callable
    from pathlib import Path
    from whisper_daemon import supervisor
    from whisper_daemon.config import Settings

    tree = ast.parse(Path("src/whisper_daemon/menubar.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MenuBarDelegate")
    function = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_update_settings"
    )
    saved = []
    scope = {
        "Callable": Callable,
        "Settings": Settings,
        "save_settings": lambda settings: saved.append(settings.capture_mic),
        "supervisor": supervisor,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "menubar.py", "exec"), scope)

    settings = Settings()
    obj = SimpleNamespace(
        _settings=settings,
        _settings_lock=threading.Lock(),
        _daemon=SimpleNamespace(_settings=settings),
    )
    mutation_entered = threading.Event()
    allow_persist = threading.Event()

    def update(current):
        current.capture_mic = False
        mutation_entered.set()
        assert allow_persist.wait(2)

    worker = threading.Thread(target=scope["_update_settings"], args=(obj, update))
    worker.start()
    assert mutation_entered.wait(2)
    try:
        assert not supervisor.request_restart_if_safe("test pressure", lambda: True)
        assert saved == []
    finally:
        allow_persist.set()
        worker.join(2)

    assert not worker.is_alive()
    assert saved == [False]


def test_file_job_is_busy_before_thread_start_and_clears_in_finally():
    import ast
    import threading
    from pathlib import Path
    from types import MethodType

    tree = ast.parse(Path("src/whisper_daemon/menubar.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MenuBarDelegate")
    functions = {
        n.name: n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_launch_file_transcription", "_run_file_transcription"}
    }
    observed = []
    obj = SimpleNamespace(
        _file_jobs=0,
        _file_jobs_lock=threading.Lock(),
        _transcribe_paths_worker=lambda paths: observed.append(("work", paths)),
    )

    class ImmediateThread:
        def __init__(self, *, target, args, **kwargs):
            self.target = target
            self.args = args

        def start(self):
            observed.append(("at-start", obj._file_jobs))
            self.target(*self.args)

    from whisper_daemon import supervisor

    scope = {
        "threading": SimpleNamespace(Thread=ImmediateThread),
        "supervisor": supervisor,
    }
    for function in functions.values():
        exec(compile(ast.Module(body=[function], type_ignores=[]), "menubar.py", "exec"), scope)
    obj._run_file_transcription = MethodType(scope["_run_file_transcription"], obj)

    scope["_launch_file_transcription"](obj, ["meeting.wav"])

    assert observed == [("at-start", 1), ("work", ["meeting.wav"])]
    assert obj._file_jobs == 0


def test_restart_rechecks_idle_inside_the_activity_gate(monkeypatch):
    from whisper_daemon import supervisor
    from whisper_daemon.memory_guard import MemoryGuard

    # First check reclaims.  On the second check work begins after the guard's
    # initial idle sample but before its restart callback obtains the gate.  A
    # later fully-idle check may restart.
    probes = iter([True, True, False, True, True])
    actions = []
    is_idle = lambda: next(probes)
    monkeypatch.setattr(
        supervisor,
        "request_restart",
        lambda reason: actions.append(("restart", reason)),
    )
    guard = MemoryGuard(
        measure=lambda: 5 * GIB,
        is_idle=is_idle,
        reclaim=lambda: actions.append("reclaim"),
        restart=lambda reason: supervisor.request_restart_if_safe(reason, is_idle),
        limit_bytes=4 * GIB,
        startup_grace_sec=0,
    )

    guard.check()
    guard.check()
    assert actions == ["reclaim"]

    guard.check()
    assert actions[1][0] == "restart"
