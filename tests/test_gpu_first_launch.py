"""Tests for the first-launch GPU detection + install prompt (#99).

Covers `should_prompt()` (pure logic — every skip branch + the show
case) and the QSettings persistence for 'Don't Ask Again'. The UI
modal in `maybe_prompt` itself is not unit-tested — it's a thin
QMessageBox that needs a Qt event loop and is exercised manually.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from forza_abyss_painter.gui import gpu_first_launch as flp
from forza_abyss_painter.gui.gpu_first_launch import (
    GpuPromptDecision, should_prompt,
)
from forza_abyss_painter.runtime.nvidia_smi import ProbeResult


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path, monkeypatch):
    """Steer QSettings into a temp file so test runs don't bleed into
    the user's real settings (and so each test starts clean)."""
    from PySide6.QtCore import QSettings, QCoreApplication
    # Set unique app paths per-test by hashing tmp_path into the
    # organization name. QSettings caches the path so isolating via
    # IniFormat + a tmp directory is the most robust isolation.
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    # Reset the never_ask key explicitly in case the cache persisted.
    s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
    s.beginGroup("gpu_first_launch")
    s.remove("never_ask")
    s.endGroup()
    s.sync()
    yield


# ==================================================== skip branches


def test_skipped_when_feature_flag_disabled(monkeypatch):
    """No GPU UX at all when the phase-3 flag is off — don't prompt."""
    monkeypatch.setattr(
        "forza_abyss_painter.gui.feature_flags.GPU_PHASE_3_AVAILABLE", False,
    )
    should, reason, probe = should_prompt()
    assert not should
    assert reason is GpuPromptDecision.SKIPPED_FLAG_DISABLED
    assert probe is None


def test_skipped_when_runtime_already_installed(monkeypatch):
    """Already installed → no point asking. Don't probe nvidia-smi
    either — pure 'check the marker' fast path."""
    monkeypatch.setattr(
        "forza_abyss_painter.gui.feature_flags.GPU_PHASE_3_AVAILABLE", True,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_installer.is_runtime_installed",
        lambda: True,
    )
    # Spy on probe — shouldn't be called.
    called = []
    monkeypatch.setattr(flp, "probe_free_vram",
                        lambda *a, **k: called.append(1) or ProbeResult(False))
    should, reason, probe = should_prompt()
    assert not should
    assert reason is GpuPromptDecision.SKIPPED_ALREADY_INSTALLED
    assert called == [], "should_prompt invoked probe despite runtime installed"


def test_skipped_when_user_opted_out(monkeypatch):
    """User clicked 'Don't ask again' previously — respect that."""
    monkeypatch.setattr(
        "forza_abyss_painter.gui.feature_flags.GPU_PHASE_3_AVAILABLE", True,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_installer.is_runtime_installed",
        lambda: False,
    )
    flp.set_user_opted_out()
    should, reason, _ = should_prompt()
    assert not should
    assert reason is GpuPromptDecision.SKIPPED_USER_OPTED_OUT


def test_skipped_when_no_nvidia_gpu(monkeypatch):
    """AMD / Intel / macOS / WSL-without-passthrough — don't nag users
    who can't use the GPU path at all."""
    monkeypatch.setattr(
        "forza_abyss_painter.gui.feature_flags.GPU_PHASE_3_AVAILABLE", True,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_installer.is_runtime_installed",
        lambda: False,
    )
    no_gpu = ProbeResult(available=False, reason="nvidia-smi not on PATH")
    monkeypatch.setattr(flp, "probe_free_vram", lambda *a, **k: no_gpu)
    should, reason, probe = should_prompt()
    assert not should
    assert reason is GpuPromptDecision.SKIPPED_NO_GPU
    assert probe is no_gpu


# ==================================================== show case


def test_should_prompt_when_gpu_present_and_no_install(monkeypatch):
    """The happy path: GPU detected, runtime not installed, user
    hasn't opted out. Should fire the prompt."""
    monkeypatch.setattr(
        "forza_abyss_painter.gui.feature_flags.GPU_PHASE_3_AVAILABLE", True,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_installer.is_runtime_installed",
        lambda: False,
    )
    gpu = ProbeResult(
        available=True, free_mib=95000, total_mib=102400,
        name="NVIDIA RTX PRO 6000 Blackwell", driver_version="555.85",
    )
    monkeypatch.setattr(flp, "probe_free_vram", lambda *a, **k: gpu)
    should, reason, probe = should_prompt()
    assert should
    # When should=True, reason is the 'placeholder' INSTALL_NOW which
    # callers ignore in favor of the actual user choice from the modal.
    assert probe is gpu


# ==================================================== persistence


def test_never_ask_default_is_false():
    """Fresh QSettings → user has NOT opted out. Critical: if this
    flipped, every user would be silently opted out on first launch."""
    assert flp.user_opted_out() is False


def test_set_user_opted_out_persists():
    flp.set_user_opted_out()
    assert flp.user_opted_out() is True


def test_set_user_opted_out_is_idempotent():
    """Calling twice doesn't break anything."""
    flp.set_user_opted_out()
    flp.set_user_opted_out()
    assert flp.user_opted_out() is True
