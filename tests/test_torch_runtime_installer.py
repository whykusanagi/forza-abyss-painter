"""Tests for forza_abyss_painter.runtime.torch_installer — the on-demand
GPU runtime installer scaffolding.

Covers everything EXCEPT the actual network downloads (which are integration-
tested manually on Windows when the EXE is rebuilt). The install pipeline is
broken into testable units: path resolution, marker serialization, install-
presence detection. The actual `install_runtime()` HTTP machinery comes in
Phase 2 of the EXE GPU bundle work; these tests pin the contracts the GUI
side will rely on.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from forza_abyss_painter.runtime import torch_installer as ti


def test_runtime_root_always_returns_writable_directory(tmp_path, monkeypatch):
    """Whatever platform, runtime_root() must return an existing, writable
    dir. The installer's first action is to write into it; if root() can
    raise or return read-only, the GUI's "first install" prompt errors
    before showing any progress."""
    # Force a temp HOME so we don't pollute the real user dir during tests.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = ti.runtime_root()
    assert root.exists()
    assert root.is_dir()
    probe = root / "write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        assert probe.read_text(encoding="utf-8") == "ok"
    finally:
        probe.unlink(missing_ok=True)


def test_runtime_root_uses_per_platform_convention(tmp_path, monkeypatch):
    """Sanity: Windows uses %LOCALAPPDATA%, macOS uses Library/Application
    Support, Linux uses XDG_DATA_HOME. Each platform-appropriate parent
    must appear in the path."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = str(ti.runtime_root())
    assert "ForzaAbyssPainter" in root
    assert "runtime" in root
    if sys.platform == "win32":
        assert "appdata" in root.lower()
    elif sys.platform == "darwin":
        assert "Library/Application Support" in root
    else:
        assert "xdg" in root or "share" in root


def test_embedded_python_paths_are_under_runtime_root(tmp_path, monkeypatch):
    """embedded_python_dir() and embedded_python_exe() must live INSIDE
    runtime_root() so deleting the runtime dir cleans up everything we
    wrote (no orphaned files anywhere else on the system)."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = ti.runtime_root()
    assert ti.embedded_python_dir().is_relative_to(root)
    assert ti.embedded_python_exe().is_relative_to(root)


def test_runtime_info_round_trips_through_marker_json(tmp_path, monkeypatch):
    """RuntimeInfo dataclass <-> JSON dict roundtrip must be lossless. The
    install pipeline writes the marker on success; the GUI reads it on
    startup. Drift here = GUI silently treats a good install as broken."""
    info = ti.RuntimeInfo(
        python_version="3.11.9",
        torch_version="2.4.1+cu121",
        cuda_available=True,
        cuda_device_name="NVIDIA GeForce RTX 4090",
        installed_at_utc="2026-05-25T12:34:56Z",
    )
    d = info.to_dict()
    assert d["torch_version"] == "2.4.1+cu121"
    assert d["cuda_available"] is True
    roundtripped = ti.RuntimeInfo.from_dict(d)
    assert roundtripped == info


def test_installed_runtime_info_returns_none_when_marker_missing(tmp_path, monkeypatch):
    """No marker file → no install. GUI shows "Install GPU runtime" button."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert ti.installed_runtime_info() is None


def test_installed_runtime_info_returns_none_when_python_exe_deleted(tmp_path, monkeypatch):
    """User deleted runtime dir but marker survived (eg manual cleanup
    that missed the file). Treat as no install — don't trust a marker
    that points at nothing."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    # Write a valid marker but DON'T create the python.exe alongside.
    info = ti.RuntimeInfo("3.11.9", "2.4.1+cu121", True, "RTX 4090",
                          "2026-05-25T12:34:56Z")
    ti.runtime_marker().write_text(json.dumps(info.to_dict()), encoding="utf-8")
    assert not ti.embedded_python_exe().exists()
    assert ti.installed_runtime_info() is None


def test_installed_runtime_info_returns_none_when_marker_corrupt(tmp_path, monkeypatch):
    """Marker file exists but is garbage / partial write / not JSON.
    Treat as no install — installer can re-run cleanly. Must NOT raise
    JSONDecodeError up to the GUI."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    ti.runtime_marker().write_text("{this is not valid json", encoding="utf-8")
    assert ti.installed_runtime_info() is None


def test_is_runtime_installed_requires_cuda_available_true(tmp_path, monkeypatch):
    """Even with a present marker + present python.exe, if cuda_available
    is False the runtime is treated as not-installed. A partial install
    (eg torch landed but no GPU detected during verify) shouldn't let the
    GUI offer "Generate locally" — would just hit torch's "No CUDA device"
    error immediately on run."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    # Create a fake python.exe so the binary-presence gate passes.
    ti.embedded_python_dir().mkdir(parents=True, exist_ok=True)
    ti.embedded_python_exe().write_text("#!/bin/sh\necho fake", encoding="utf-8")
    # Marker says CUDA is NOT available.
    info = ti.RuntimeInfo("3.11.9", "2.4.1", False, "",
                          "2026-05-25T12:34:56Z")
    ti.runtime_marker().write_text(json.dumps(info.to_dict()), encoding="utf-8")
    assert ti.installed_runtime_info() is not None    # marker parses
    assert ti.is_runtime_installed() is False         # but is_installed gates on CUDA


def test_is_runtime_installed_true_with_full_install(tmp_path, monkeypatch):
    """All three conditions met (marker exists, parses, cuda_available=True,
    python.exe present) → is_runtime_installed returns True. GUI shows
    "Generate locally" button enabled."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    ti.embedded_python_dir().mkdir(parents=True, exist_ok=True)
    ti.embedded_python_exe().write_text("#!/bin/sh\necho fake", encoding="utf-8")
    info = ti.RuntimeInfo("3.11.9", "2.4.1+cu121", True, "RTX 4090",
                          "2026-05-25T12:34:56Z")
    ti.runtime_marker().write_text(json.dumps(info.to_dict()), encoding="utf-8")
    assert ti.is_runtime_installed() is True


def test_estimated_download_bytes_is_realistic():
    """~4 GiB total. If this drifts (eg torch wheel shrinks dramatically
    or we pick a smaller index), revisit the GUI's confirmation prompt
    to reduce friction."""
    gib = ti.estimated_download_bytes() / (1 << 30)
    assert 2.0 <= gib <= 10.0, f"estimated download {gib:.1f} GiB outside sane range"


def test_pinned_versions_are_documented():
    """Version pins are part of the public install contract — if they
    change silently, users' existing installs become stale + the GUI
    needs to know to re-install. Test pins exist as module constants."""
    assert ti.EMBED_PYTHON_VERSION.startswith("3.11")
    assert ti.TORCH_VERSION.startswith("2.")
    assert "cu" in ti.TORCH_CUDA_INDEX
