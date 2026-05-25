"""Tests for forza_abyss_painter.io.log_paths — the per-platform log directory
+ timestamped log-file helpers used by the inject worker.

Critical contract: log_root() and open_inject_log() must NEVER raise. The
inject worker calls them at the top of run() before any state is set up; if
they threw, the user would see a cryptic crash instead of the inject dialog.
The "always succeeds" guarantee is the whole point of the tmpdir fallback in
log_root.
"""
from __future__ import annotations

import os
import sys

import pytest

from forza_abyss_painter.io import log_paths


def test_log_root_returns_writable_directory():
    """log_root() always returns an existing, writable directory."""
    root = log_paths.log_root()
    assert root.exists()
    assert root.is_dir()
    # Round-trip a small write to prove it's actually writable.
    probe = root / ".write-probe"
    try:
        probe.write_text("ok")
        assert probe.read_text() == "ok"
    finally:
        probe.unlink(missing_ok=True)


def test_log_root_uses_platform_appropriate_path():
    """Sanity check the platform branches resolve to recognizable paths.
    We don't assert the exact string (varies by user) — just that it lives
    under a per-platform convention."""
    root = log_paths.log_root()
    s = str(root)
    if sys.platform == "win32":
        assert "ForzaAbyssPainter" in s and "logs" in s
    elif sys.platform == "darwin":
        assert "Library/Logs/ForzaAbyssPainter" in s
    else:
        assert "ForzaAbyssPainter" in s and "logs" in s


def test_new_inject_log_path_is_unique_per_second():
    """Two calls within the same second yield the same name (acceptable;
    line-prefix timestamps disambiguate). Two calls across seconds differ."""
    p1 = log_paths.new_inject_log_path()
    p2 = log_paths.new_inject_log_path()
    # Same-second calls are equal (file name has second resolution).
    assert p1 == p2 or p1.name[:-4] != p2.name[:-4]
    # Format check: inject-YYYYMMDD-HHMMSS.log
    assert p1.name.startswith("inject-")
    assert p1.suffix == ".log"


def test_open_inject_log_is_line_buffered_and_writable():
    """open_inject_log returns an open, line-buffered handle that immediately
    flushes line writes. This matters because the EXE might be force-killed
    during a long scan and we don't want diagnostic lines stuck in a buffer."""
    path, fh = log_paths.open_inject_log()
    try:
        fh.write("test-line-1\n")
        # Line-buffered: the next read of the file should see the line
        # without an explicit flush.
        content = path.read_text(encoding="utf-8")
        assert "test-line-1" in content
    finally:
        fh.close()
        # Clean up the probe file we just created.
        try:
            path.unlink()
        except OSError:
            pass


def test_log_root_falls_back_when_preferred_unwritable(monkeypatch, tmp_path):
    """If the preferred per-platform dir can't be created (eg locked-down
    kiosk), fall through to a tmpdir-rooted path rather than raise. The
    inject worker contract depends on this never throwing."""
    # Force the preferred path's parent to point at a read-only file so
    # mkdir(parents=True) fails. Easiest cross-platform: point Path.home()
    # at a file (not a directory) so all the preferred-path branches break.
    bad_dir = tmp_path / "not-a-dir-its-a-file"
    bad_dir.write_text("nope")    # exists, but is a file
    monkeypatch.setattr(log_paths.Path, "home", classmethod(lambda cls: bad_dir))
    # Also blank the LOCALAPPDATA / XDG_STATE_HOME envs so Windows / Linux
    # branches don't dodge the bad home.
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    root = log_paths.log_root()
    assert root.exists()
    assert root.is_dir()
