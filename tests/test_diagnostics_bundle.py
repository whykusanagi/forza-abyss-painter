"""Tests for forza_abyss_painter.runtime.diagnostics.

Covers:
  - build_bundle writes a valid zip with the expected layout (README,
    logs/, runtime/installed.json if present, system_info.json)
  - Empty logs dir + no marker still produces a valid bundle (the
    bundle is useful even when there's nothing to report)
  - include_runtime_dir flag adds the embedded Python tree
  - fap-diag CLI: --output, default path, exit codes
  - _system_info captures the contract fields
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forza_abyss_painter.runtime import diagnostics as diag
from forza_abyss_painter.runtime import gpu_logger as gl
from forza_abyss_painter.runtime import torch_installer as ti


@pytest.fixture
def _isolated_dirs(tmp_path, monkeypatch):
    """Redirect logs_root + runtime_root so the bundle reads from a
    sandbox instead of the user's real LOCALAPPDATA / Library / XDG."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(diag.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    gl.reset_gpu_logger_for_tests()
    yield tmp_path
    gl.reset_gpu_logger_for_tests()


# ====================================================================
# build_bundle: structure


def test_bundle_contains_readme_at_root(_isolated_dirs):
    """README.md is the first thing a recipient sees on extract. If it
    drifts off, support requests come without the context the doc
    provides about WHAT each file is."""
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        assert "README.md" in zf.namelist()
        readme = zf.read("README.md").decode("utf-8")
    assert "Forza Abyss Painter" in readme
    assert "logs/" in readme


def test_bundle_includes_existing_log_files(_isolated_dirs):
    """Every .log file in logs_root() lands in logs/<filename>. Without
    this the bundle is empty of the most important data."""
    # Create a couple of fake session logs.
    log_dir = gl.logs_root()
    (log_dir / "gpu-2026-05-25-001-main.log").write_text(
        '{"kind":"session_start","ts":"2026-05-25T00:00:00Z"}\n',
        encoding="utf-8",
    )
    (log_dir / "gpu-2026-05-25-001-runner.log").write_text(
        '{"kind":"session_start","ts":"2026-05-25T00:00:01Z"}\n',
        encoding="utf-8",
    )
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "logs/gpu-2026-05-25-001-main.log" in names
        assert "logs/gpu-2026-05-25-001-runner.log" in names


def test_bundle_includes_install_marker_when_present(_isolated_dirs):
    """Install marker → runtime/installed.json. Capturing the marker
    is critical: it tells the maintainer what torch version + CUDA
    state the runtime claimed when the user hit a problem."""
    # Write a fake install marker.
    info = ti.RuntimeInfo(
        python_version="3.11.9", torch_version="2.4.1+cu121",
        cuda_available=True, cuda_device_name="RTX 4090",
        installed_at_utc="2026-05-25T12:00:00Z",
    )
    ti.runtime_marker().write_text(json.dumps(info.to_dict()), encoding="utf-8")
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        assert "runtime/installed.json" in zf.namelist()
        marker_data = json.loads(zf.read("runtime/installed.json"))
    assert marker_data["torch_version"] == "2.4.1+cu121"
    assert marker_data["cuda_device_name"] == "RTX 4090"


def test_bundle_handles_missing_marker_cleanly(_isolated_dirs):
    """No install attempted yet → no marker → bundle still builds. The
    user might run fap-diag BEFORE attempting install to capture system
    state pre-emptively. Skipping the marker on absence is the right
    behavior (vs. raising)."""
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "runtime/installed.json" not in names
        # README + system_info still landed.
        assert "README.md" in names
        assert "system_info.json" in names


def test_bundle_always_includes_system_info(_isolated_dirs):
    """system_info.json captures OS + Python + nvidia-smi state at
    bundle time. Even without logs, this is enough for the maintainer
    to know what kind of machine the tester is running on."""
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        info = json.loads(zf.read("system_info.json"))
    assert "platform" in info
    assert "python_version" in info
    assert "captured_at_utc" in info


def test_bundle_excludes_runtime_dir_by_default(_isolated_dirs):
    """Default bundle stays emailable (~MiB, not GiB). The embedded
    Python tree is only included when explicitly requested."""
    # Create a fake runtime dir with a giant file.
    rt = ti.runtime_root()
    embed_dir = ti.embedded_python_dir()
    embed_dir.mkdir(parents=True, exist_ok=True)
    (embed_dir / "fake_torch_wheel.dylib").write_bytes(b"x" * 1024)
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out, include_runtime_dir=False)
    with zipfile.ZipFile(out) as zf:
        assert not any("python311" in name for name in zf.namelist()), (
            "runtime dir contents leaked into default bundle"
        )


def test_bundle_includes_runtime_dir_when_requested(_isolated_dirs):
    """--include-runtime-dir flag includes the embedded Python tree.
    Used when install artifacts themselves are suspect."""
    embed_dir = ti.embedded_python_dir()
    embed_dir.mkdir(parents=True, exist_ok=True)
    (embed_dir / "python_marker.txt").write_text("hi", encoding="utf-8")
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out, include_runtime_dir=True)
    with zipfile.ZipFile(out) as zf:
        assert any("python_marker.txt" in name for name in zf.namelist()), (
            "include_runtime_dir=True didn't include runtime tree contents"
        )


def test_bundle_is_a_valid_zip(_isolated_dirs):
    """The output is a real zip. Apparently obvious but defensive
    against future refactors that might use a different format."""
    out = _isolated_dirs / "bundle.zip"
    diag.build_bundle(out)
    assert zipfile.is_zipfile(out)


# ====================================================================
# CLI


def test_main_with_default_output_writes_to_home(_isolated_dirs, monkeypatch, capsys):
    """No --output: write to ~/ForzaAbyssPainter-diag-{ts}.zip. The
    test fixture's monkeypatched home points at tmp, so we can verify
    the file lands there."""
    rc = diag.main([])
    assert rc == 0
    captured = capsys.readouterr()
    # Stdout includes the absolute path so user knows where the zip went.
    assert "ForzaAbyssPainter-diag-" in captured.out
    assert ".zip" in captured.out


def test_main_with_explicit_output(_isolated_dirs, capsys):
    """Explicit --output overrides the default path. Used by the GUI
    menu hook to direct the bundle to a user-picked location."""
    out = _isolated_dirs / "custom_diag.zip"
    rc = diag.main(["--output", str(out)])
    assert rc == 0
    assert out.exists()


def test_main_returns_nonzero_on_filesystem_error(_isolated_dirs, monkeypatch, capsys):
    """OSError on zip write → exit code 1 with stderr message. The GUI
    can check the exit code to decide whether to surface a 'bundle
    saved' confirmation or an error modal."""
    monkeypatch.setattr(diag, "build_bundle",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    rc = diag.main([])
    assert rc == 1
    captured = capsys.readouterr()
    assert "failed" in captured.err.lower() or "disk full" in captured.err


# ====================================================================
# _system_info


def test_system_info_includes_contract_fields():
    """Required fields for triage: platform, python_version,
    captured_at_utc. Without these the system_info file is useless."""
    info = diag._system_info()
    assert "platform" in info
    assert "python_version" in info
    assert "captured_at_utc" in info
    # captured_at_utc parses as ISO 8601.
    from datetime import datetime
    datetime.fromisoformat(info["captured_at_utc"])


def test_system_info_handles_missing_nvidia_smi():
    """No nvidia-smi (macOS dev, AMD GPU, etc) → field captures the
    absence rather than crashing. Empty bundle would be confusing."""
    info = diag._system_info()
    # Either nvidia_smi (success) OR nvidia_smi_unavailable (failure)
    # is present — never neither.
    assert "nvidia_smi" in info or "nvidia_smi_unavailable" in info
