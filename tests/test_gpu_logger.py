"""Tests for forza_abyss_painter.runtime.gpu_logger.

Covers:
  - logs_root() platform-aware path resolution
  - GpuLogger writes JSONL with the contract fields (ts, kind,
    process, thread, elapsed_s)
  - start_phase context manager emits start/done on success +
    start/error on exception
  - log_exception captures class, message, traceback
  - close writes session_end event idempotently
  - Process-singleton accessor returns the same instance
  - Thread safety: concurrent log calls from N threads land N events

No torch / no GUI / no subprocess — the logger is a leaf module + must
work even when the GPU runtime is uninstalled.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from forza_abyss_painter.runtime import gpu_logger as gl


# ====================================================================
# Path resolution


def test_logs_root_uses_per_platform_convention(tmp_path, monkeypatch):
    """Sanity: Windows → %LOCALAPPDATA%, macOS → Library/Application
    Support, Linux → XDG_DATA_HOME. Each platform-appropriate parent
    must appear in the path so tester-supplied logs are findable on
    their box without grepping the whole filesystem."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = str(gl.logs_root())
    assert "ForzaAbyssPainter" in root
    assert "logs" in root
    if sys.platform == "win32":
        assert "appdata" in root.lower()
    elif sys.platform == "darwin":
        assert "Library/Application Support" in root
    else:
        assert "xdg" in root or "share" in root


def test_logs_root_creates_directory(tmp_path, monkeypatch):
    """First call must create the directory tree. If it raises (e.g.,
    permissions), the EXE's GPU path can't log anything — diagnostics
    are dead on arrival."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = gl.logs_root()
    assert root.exists() and root.is_dir()


def test_session_log_path_distinguishes_main_vs_runner(tmp_path, monkeypatch):
    """The main GUI process + the torch_runner subprocess each log to
    their own file in the same dir. Diagnostics bundle ships both. If
    they collided on filename the tester would only get half the
    picture."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    main_path = gl.session_log_path("main")
    runner_path = gl.session_log_path("runner")
    assert main_path != runner_path
    assert main_path.parent == runner_path.parent
    assert "main" in main_path.name
    assert "runner" in runner_path.name


# ====================================================================
# Event format


@pytest.fixture
def _isolated_log(tmp_path, monkeypatch):
    """Redirect logs_root → tmp dir so tests don't pollute real
    LOCALAPPDATA. Yields the redirected root."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    gl.reset_gpu_logger_for_tests()
    yield gl.logs_root()
    gl.reset_gpu_logger_for_tests()


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_logger_writes_session_start_event_on_construction(_isolated_log):
    """A fresh logger immediately writes a session_start event so even
    a crash 1 microsecond later leaves SOMETHING in the file. Without
    this, an early-init crash looks identical to 'EXE never launched'
    when looking at the empty logs dir."""
    logger = gl.GpuLogger()
    events = _read_events(logger.session_path)
    assert any(e["kind"] == "session_start" for e in events)
    start = [e for e in events if e["kind"] == "session_start"][0]
    assert start["process"] == "main"
    assert "python_version" in start
    assert "platform" in start


def test_log_event_includes_contract_fields(_isolated_log):
    """Every event MUST carry ts, kind, process, thread, elapsed_s.
    Downstream log analysis (post-mortem scripts, diagnostics bundle
    parser) depends on this contract. Missing fields = broken triage."""
    logger = gl.GpuLogger()
    logger.log("test_event", custom_field=42)
    events = _read_events(logger.session_path)
    test_events = [e for e in events if e["kind"] == "test_event"]
    assert len(test_events) == 1
    e = test_events[0]
    for required in ("ts", "kind", "process", "thread", "elapsed_s"):
        assert required in e, f"event missing required field {required!r}: {e}"
    # Custom field preserved.
    assert e["custom_field"] == 42


def test_log_event_serializes_non_json_values_safely(_isolated_log):
    """Pass a Path or a class as a field value — log() must not crash.
    Defensive JSON serialization via default=str handles types that
    have a sensible __str__ but aren't JSON-native."""
    logger = gl.GpuLogger()
    logger.log("path_event", path=Path("/tmp/x"), cls=int)
    events = _read_events(logger.session_path)
    assert any(e["kind"] == "path_event" for e in events)


# ====================================================================
# start_phase context manager


def test_start_phase_emits_start_and_done_on_success(_isolated_log):
    """Happy path: phase_start at enter + phase_done at clean exit.
    phase_done carries phase_elapsed_s for performance triage."""
    logger = gl.GpuLogger()
    with logger.start_phase("download_python", url="https://example/x.zip"):
        pass
    events = _read_events(logger.session_path)
    starts = [e for e in events if e["kind"] == "phase_start"]
    dones = [e for e in events if e["kind"] == "phase_done"]
    assert len(starts) == 1 and starts[0]["stage"] == "download_python"
    assert starts[0]["url"] == "https://example/x.zip"
    assert len(dones) == 1 and dones[0]["stage"] == "download_python"
    assert "phase_elapsed_s" in dones[0]


def test_start_phase_emits_error_event_on_exception(_isolated_log):
    """Exception during phase → phase_error event with type, message,
    traceback before re-raising. The user's caller still sees the
    exception, but the log captured enough to reproduce."""
    logger = gl.GpuLogger()
    class CustomError(Exception):
        pass
    with pytest.raises(CustomError):
        with logger.start_phase("pip_install"):
            raise CustomError("pip exit 1")
    events = _read_events(logger.session_path)
    errors = [e for e in events if e["kind"] == "phase_error"]
    assert len(errors) == 1
    err = errors[0]
    assert err["stage"] == "pip_install"
    assert err["exception_class"] == "CustomError"
    assert "pip exit 1" in err["exception_message"]
    assert "Traceback" in err["traceback"] or "CustomError" in err["traceback"]


def test_log_exception_captures_traceback(_isolated_log):
    """Direct log_exception() call (not via context manager) also
    captures the traceback string. Used in except: blocks that don't
    fit the start_phase pattern."""
    logger = gl.GpuLogger()
    try:
        raise ValueError("test exc")
    except ValueError as exc:
        logger.log_exception("explicit_exception", exc, where="test")
    events = _read_events(logger.session_path)
    excs = [e for e in events if e["kind"] == "explicit_exception"]
    assert len(excs) == 1
    e = excs[0]
    assert e["exception_class"] == "ValueError"
    assert e["exception_message"] == "test exc"
    assert "traceback" in e
    assert e["where"] == "test"


# ====================================================================
# Session close


def test_close_writes_session_end_with_outcome(_isolated_log):
    """close() appends session_end. Without an explicit end event we
    can't distinguish 'crashed mid-run' from 'completed normally',
    which matters when triaging a Windows tester's logs."""
    logger = gl.GpuLogger()
    logger.log("midway")
    logger.close(outcome="manual_close")
    events = _read_events(logger.session_path)
    assert events[-1]["kind"] == "session_end"
    assert events[-1]["outcome"] == "manual_close"
    assert "total_elapsed_s" in events[-1]


def test_close_is_idempotent(_isolated_log):
    """A second close() is a no-op — atexit can race with explicit
    close calls. If close emitted a duplicate session_end, log parsers
    would double-count or get confused about WHICH end was real."""
    logger = gl.GpuLogger()
    logger.close(outcome="first")
    logger.close(outcome="second")
    events = _read_events(logger.session_path)
    ends = [e for e in events if e["kind"] == "session_end"]
    assert len(ends) == 1, f"close not idempotent — got {len(ends)} session_end events"


# ====================================================================
# Singleton accessor


def test_get_gpu_logger_returns_same_instance(_isolated_log):
    """Process-singleton accessor. All GPU code paths (install worker,
    gen worker, dialogs, subprocess) must log to the SAME file in this
    process — different files would scatter the diagnostic trail."""
    a = gl.get_gpu_logger()
    b = gl.get_gpu_logger()
    assert a is b


def test_get_gpu_logger_reset_creates_new_instance(_isolated_log):
    """Test helper: reset_gpu_logger_for_tests drops the singleton so
    each test gets a clean slate. Without this the first test's events
    bleed into the second test's assertions."""
    a = gl.get_gpu_logger()
    gl.reset_gpu_logger_for_tests()
    b = gl.get_gpu_logger()
    assert a is not b


# ====================================================================
# Thread safety


def test_concurrent_log_calls_dont_lose_events(_isolated_log):
    """N threads each writing M events → N*M events end up in the file,
    each one a valid JSON line. The writer lock serializes; without it,
    interleaved partial writes corrupt the JSONL format and parsers
    error on the next line break."""
    logger = gl.GpuLogger()
    NUM_THREADS = 8
    EVENTS_PER_THREAD = 25
    barrier = threading.Barrier(NUM_THREADS)

    def _writer(tid):
        barrier.wait()
        for i in range(EVENTS_PER_THREAD):
            logger.log("concurrent_event", tid=tid, i=i)

    threads = [threading.Thread(target=_writer, args=(t,))
               for t in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = _read_events(logger.session_path)
    concurrent = [e for e in events if e["kind"] == "concurrent_event"]
    assert len(concurrent) == NUM_THREADS * EVENTS_PER_THREAD, (
        f"expected {NUM_THREADS * EVENTS_PER_THREAD} concurrent events, "
        f"got {len(concurrent)} — writer lock not effective or events lost"
    )
    # Every event has a valid (tid, i) pair → no corrupted lines.
    pairs = {(e["tid"], e["i"]) for e in concurrent}
    assert len(pairs) == NUM_THREADS * EVENTS_PER_THREAD, (
        "duplicate or missing (tid, i) pairs — events corrupted"
    )
