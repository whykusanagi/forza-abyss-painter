"""Per-platform log directory + timestamped log-file helpers.

Used by the injection worker (and any future EXE-side telemetry) to persist a
copy of every status_cb line to disk. Without this the [fast-locate] /
[readiness] diagnostics live only in the injection dialog and vanish the
moment the user closes it — which made debugging the first round of fast-mode
misses impossible.

Platforms:
  Windows  : %LOCALAPPDATA%\\ForzaAbyssPainter\\logs\\
  macOS    : ~/Library/Logs/ForzaAbyssPainter/
  Linux/*  : $XDG_STATE_HOME/ForzaAbyssPainter/logs/ (falls back to ~/.local/state/…)

We hand-roll these instead of pulling in platformdirs to keep the PyInstaller
bundle a few hundred kB smaller and avoid one more import surface.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


APP_NAME = "ForzaAbyssPainter"


def log_root() -> Path:
    """Return the per-user log directory for this app, creating it if missing.

    Always succeeds — falls back to a tmpdir-style path if the preferred
    location can't be created (eg locked-down kiosk system) so logging never
    silently disables itself.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        root = base / APP_NAME / "logs"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs" / APP_NAME
    else:
        # Linux / BSD — XDG state dir.
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
        root = base / APP_NAME / "logs"
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError:
        # Last-resort fallback so logging.basicConfig never blows up.
        import tempfile
        fallback = Path(tempfile.gettempdir()) / APP_NAME / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def new_inject_log_path() -> Path:
    """Return a fresh log path of the form <root>/inject-YYYYMMDD-HHMMSS.log.

    UTC timestamp so logs from different timezones sort correctly when shared.
    Uniqueness across rapid clicks isn't guaranteed (per-second resolution); if
    two injections start in the same second the second one's writes append to
    the first one's file — harmless because the worker prefixes every line
    with its own ISO timestamp.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return log_root() / f"inject-{stamp}.log"


def open_inject_log() -> tuple[Path, object]:
    """Open a fresh inject log for append-writing. Returns (path, file_handle).

    File handle is line-buffered (`buffering=1`) so each status line flushes
    immediately — important because the EXE might be force-killed during a
    long scan and we don't want the diagnostic lines stuck in a 4 KiB buffer.
    """
    path = new_inject_log_path()
    fh = path.open("a", encoding="utf-8", buffering=1)
    return path, fh
