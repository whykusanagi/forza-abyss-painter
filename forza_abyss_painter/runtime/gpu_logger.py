"""Structured per-session diagnostic logger for the EXE's GPU pipeline.

Writes one JSON object per line to
`%LOCALAPPDATA%/ForzaAbyssPainter/logs/gpu-YYYY-MM-DD-HHMMSS.log` (mirror
paths on macOS/Linux for testing). Both the main EXE process AND the
torch_runner subprocess write to the same logs directory — different
file names, same parent dir, so the diagnostics-bundle export ships
the full picture.

Why this matters: the GPU pipeline runs across a process boundary (main
GUI + embedded-Python subprocess) and a thread boundary (Qt workers).
When a Windows tester reports "it didn't work" we need to know whether
the install failed (and which phase), the runner subprocess crashed
(and where), the IPC parsed wrong, or the engine ran fine but produced
zero shapes. Generic 'check the logs' is useless if there are no logs;
structured logs that name WHICH phase WHERE + WHEN + WHAT exception is
the difference between fixing it in 5 minutes and a debugging marathon.

## API

  get_gpu_logger() -> GpuLogger
      Returns the process-singleton logger, lazy-creating the session
      file on first use.

  logger.log(kind, **fields)
      Append one event. `kind` is a short string identifier
      ("phase_start", "phase_done", "phase_error", "subprocess_spawn",
      "signal_received", "session_end", etc). `fields` is freeform JSON.

  logger.log_exception(kind, exc, **fields)
      Append event with type, message, traceback string. For except:
      branches.

  logger.start_phase(stage) -> ctx manager
      Emits phase_start on enter, phase_done on clean exit, phase_error
      on exception. Elapsed seconds included on done/error.

  logger.session_path -> Path
      The JSONL file this logger is writing to. The diagnostics-bundle
      export reads this to know which file to include.

## Event schema

  {"ts": "2026-05-25T12:34:56.789Z",
   "process": "main" | "runner",
   "thread": "MainThread" | <thread name>,
   "elapsed_s": <float from logger init>,
   "kind": "<event kind>",
   ... freeform fields ...}

## Cross-process correlation

Each process logs its own session file. The session file name carries
the timestamp of process start, so correlating "what happened" is just
sorting by filename + reading in order. The main-process logger emits
a `subprocess_spawn` event with the expected subprocess log filename
when it spawns torch_runner, so the bundle can show "this subprocess
event came from that run".
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback as tb_mod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# Determined once per process so all events in a session use the same
# millisecond timestamp prefix in their filename → easy filesystem sort.
_SESSION_TS = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S-%f")[:-3]


def logs_root() -> Path:
    """Per-user logs directory. Created if missing.

    Windows: %LOCALAPPDATA%/ForzaAbyssPainter/logs/
    macOS:   ~/Library/Application Support/ForzaAbyssPainter/logs/
    Linux:   $XDG_DATA_HOME/ForzaAbyssPainter/logs/

    Mirrors the runtime_root() convention so logs land next to the
    runtime install — easy for users to find both at once.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    root = base / "ForzaAbyssPainter" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_log_path(process_label: str = "main") -> Path:
    """Filename for THIS process's session log. process_label
    distinguishes main from runner subprocess so a single tester's
    bundle has both halves visible side-by-side."""
    return logs_root() / f"gpu-{_SESSION_TS}-{process_label}.log"


class GpuLogger:
    """Per-process structured logger. Thread-safe (the writer lock
    serializes appends across Qt worker threads + the GUI thread).

    Don't instantiate directly — use get_gpu_logger() so the whole
    process shares one log file. The constructor is public mainly
    for tests that need to redirect the output path.
    """

    def __init__(
        self,
        path: Path | None = None,
        process_label: str = "main",
    ) -> None:
        self.session_path: Path = path or session_log_path(process_label)
        self.process_label: str = process_label
        self._t0: float = time.monotonic()
        self._lock = threading.Lock()
        self._fp = open(self.session_path, "a", encoding="utf-8", buffering=1)
        self.log("session_start", session_ts=_SESSION_TS,
                 process_label=process_label,
                 python_version=sys.version.split()[0],
                 platform=sys.platform)

    def log(self, kind: str, **fields: Any) -> None:
        """Append one event. Captures wall-clock timestamp + elapsed
        seconds from logger init + the calling thread's name."""
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "process": self.process_label,
            "thread": threading.current_thread().name,
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            "kind": kind,
        }
        event.update(fields)
        line = json.dumps(event, default=str)
        with self._lock:
            self._fp.write(line + "\n")
            # Line-buffered open + explicit flush is belt-and-suspenders:
            # if the process crashes between buffer flush and disk sync,
            # at least the last logged line is visible to a tester
            # investigating with `tail -f`.
            self._fp.flush()

    def log_exception(self, kind: str, exc: BaseException, **fields: Any) -> None:
        """Append event with the exception's class, message, and the
        full traceback string. Use this from `except:` branches so the
        log captures enough to reproduce."""
        self.log(
            kind,
            exception_class=type(exc).__name__,
            exception_message=str(exc),
            traceback=tb_mod.format_exc(),
            **fields,
        )

    @contextmanager
    def start_phase(self, stage: str, **fields: Any) -> Iterator[None]:
        """Bracket a phase with start/done/error events. Use as a
        context manager around any meaningful unit of work — phases
        in install_runtime, sub-steps in torch_runner, etc.

            with logger.start_phase("download_python", url=URL):
                urlretrieve(...)
            # phase_done event with elapsed_s if no exception
            # phase_error event with traceback if it raised
        """
        t0 = time.monotonic()
        self.log("phase_start", stage=stage, **fields)
        try:
            yield
        except BaseException as exc:
            self.log_exception("phase_error", exc,
                               stage=stage,
                               phase_elapsed_s=round(time.monotonic() - t0, 3),
                               **fields)
            raise
        else:
            self.log("phase_done", stage=stage,
                     phase_elapsed_s=round(time.monotonic() - t0, 3),
                     **fields)

    def close(self, outcome: str = "ok", **fields: Any) -> None:
        """Write a session_end event + close the file. Idempotent — a
        second call is a no-op. Always called via atexit so even hard
        crashes leave a closed log file (the buffered write may not
        have all events, but the structure is complete enough for
        post-mortem). Production callers usually don't call this
        explicitly; the atexit hook handles it."""
        if self._fp.closed:
            return
        self.log("session_end", outcome=outcome,
                 total_elapsed_s=round(time.monotonic() - self._t0, 3),
                 **fields)
        try:
            self._fp.close()
        except OSError:
            pass


# ---------- process-singleton accessor ----------

_LOGGER: GpuLogger | None = None
_LOGGER_LOCK = threading.Lock()


def get_gpu_logger(process_label: str = "main") -> GpuLogger:
    """Return the process's shared GpuLogger. Lazy-creates on first
    call, then returns the same instance. process_label only takes
    effect on first call — subsequent calls return the existing
    logger regardless of label argument."""
    global _LOGGER
    if _LOGGER is None:
        with _LOGGER_LOCK:
            if _LOGGER is None:
                _LOGGER = GpuLogger(process_label=process_label)
                # Register atexit so the session_end event lands even
                # on hard exit (sys.exit, uncaught exception, etc).
                import atexit
                atexit.register(_LOGGER.close, outcome="atexit")
    return _LOGGER


def reset_gpu_logger_for_tests() -> None:
    """Drop the singleton so the next get_gpu_logger() call opens a
    fresh session. Test-only — production never resets."""
    global _LOGGER
    with _LOGGER_LOCK:
        if _LOGGER is not None:
            _LOGGER.close(outcome="reset_for_tests")
        _LOGGER = None
