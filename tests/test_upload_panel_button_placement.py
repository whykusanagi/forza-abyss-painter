"""The Re-shape-gen + Polish buttons must be placed BEFORE the
stack (recent/image_search) so they don't get pushed off-screen by
the stack's stretch=1 vertical expansion (user-reported regression)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.upload_panel import UploadPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_reshape_polish_buttons_are_above_the_stack(qapp):
    """The reshape/polish row must appear at a smaller layout index
    than the stack widget. Otherwise the stack's stretch=1 pushes the
    buttons off-screen on real-EXE window heights."""
    panel = UploadPanel()
    layout = panel.layout()
    # Find the index of self.stack and of the row containing reshape_btn.
    stack_index = None
    reshape_row_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is panel.stack:
            stack_index = i
            continue
        sub_layout = item.layout()
        if sub_layout is not None:
            for j in range(sub_layout.count()):
                sub_widget = sub_layout.itemAt(j).widget()
                if sub_widget is panel.reshape_btn:
                    reshape_row_index = i
                    break
    assert stack_index is not None, "self.stack not found in layout"
    assert reshape_row_index is not None, "reshape_btn row not found in layout"
    assert reshape_row_index < stack_index, (
        f"reshape_polish_row at index {reshape_row_index} must come "
        f"BEFORE self.stack at index {stack_index} — otherwise the "
        f"stack's stretch=1 pushes the buttons off-screen"
    )
    panel.deleteLater()
