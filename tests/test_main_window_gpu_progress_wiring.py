"""GPU progress signals must also update the middle PreviewPanel
(user-reported regression where the middle panel stayed on 'Idle'
during a GPU run because only the bottom status bar was wired)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.preview_panel import PreviewPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_preview_panel_on_progress_accepts_no_rms(qapp):
    """PreviewPanel.on_progress must accept (count, total) without
    rms — the GPU path doesn't have an RMS value to emit."""
    panel = PreviewPanel()
    panel.on_progress(100, 1000)
    assert panel.progress.value() == 10
    assert "100/1000" in panel.status_label.text()
    # Without rms, status label should not include the RMS suffix.
    assert "RMS" not in panel.status_label.text()
    panel.deleteLater()


def test_preview_panel_on_progress_accepts_rms(qapp):
    """CPU path still emits RMS; preview still renders it."""
    panel = PreviewPanel()
    panel.on_progress(100, 1000, rms=42.5)
    assert "RMS=42.50" in panel.status_label.text()
    panel.deleteLater()


def test_preview_panel_on_progress_rounds_pct(qapp):
    """Standard percentage math."""
    panel = PreviewPanel()
    panel.on_progress(333, 1000)
    assert panel.progress.value() == 33
    panel.deleteLater()


def test_preview_panel_on_progress_clamps_high(qapp):
    """If a checkpoint fires past total (rare but possible), the
    progress bar caps at 100."""
    panel = PreviewPanel()
    panel.on_progress(1100, 1000)
    assert panel.progress.value() == 100
    panel.deleteLater()
