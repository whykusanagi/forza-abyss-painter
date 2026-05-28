"""Resume from snapshot… button is always visible (no flag, no
loaded-JSON gate) and emits the picked path via resume_requested."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.upload_panel import UploadPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_resume_button_exists_and_visible(qapp):
    panel = UploadPanel()
    assert hasattr(panel, "resume_btn")
    # Always visible — no feature flag, no Upload-JSON gate.
    assert panel.resume_btn.isVisible() or panel.resume_btn.isVisibleTo(panel)
    panel.deleteLater()


def test_resume_button_emits_signal_with_picked_path(qapp, tmp_path):
    snap = tmp_path / "x_2900.json"
    snap.write_text("{}")
    panel = UploadPanel()
    received: list[Path] = []
    panel.resume_requested.connect(lambda p: received.append(p))
    from PySide6.QtWidgets import QFileDialog
    with patch.object(QFileDialog, "getOpenFileName",
                       return_value=(str(snap), "Forza Abyss Painter snapshots (*_*.json)")):
        panel.resume_btn.click()
    assert received == [snap]
    panel.deleteLater()


def test_resume_button_cancel_emits_nothing(qapp):
    panel = UploadPanel()
    received: list[Path] = []
    panel.resume_requested.connect(lambda p: received.append(p))
    from PySide6.QtWidgets import QFileDialog
    with patch.object(QFileDialog, "getOpenFileName",
                       return_value=("", "")):
        panel.resume_btn.click()
    assert received == []
    panel.deleteLater()


def test_resume_button_placement_below_tier_b_row(qapp):
    """Layout sanity: resume button is BELOW the Tier B reshape/polish
    row but ABOVE the Recent stack (so the stretch=1 stack doesn't
    push it offscreen)."""
    panel = UploadPanel()
    layout = panel.layout()
    stack_index = None
    reshape_row_index = None
    resume_row_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is panel.stack:
            stack_index = i
            continue
        sub_layout = item.layout()
        if sub_layout is not None:
            for j in range(sub_layout.count()):
                sw = sub_layout.itemAt(j).widget()
                if sw is panel.reshape_btn:
                    reshape_row_index = i
                if sw is panel.resume_btn:
                    resume_row_index = i
    assert stack_index is not None
    assert reshape_row_index is not None
    assert resume_row_index is not None
    assert reshape_row_index < resume_row_index < stack_index, (
        f"layout order wrong: reshape={reshape_row_index} "
        f"resume={resume_row_index} stack={stack_index}"
    )
    panel.deleteLater()
