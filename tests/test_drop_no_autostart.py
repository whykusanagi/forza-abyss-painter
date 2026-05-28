"""File drop adds to queue but must NOT auto-trigger a run.

Only settings_panel.start_clicked (Start button) should fire _start_next.
The auto-chain in _on_done (after a run completes) is preserved
separately — this test only pins the drop/upload entry point.
"""
import os
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QApplication


def _make_image(tmp_path: Path) -> Path:
    p = tmp_path / "smoke.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(p)
    return p


def test_file_drop_does_not_autostart_run(tmp_path):
    """Dropping a file adds it to the queue but must NOT call _start_next."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        img = _make_image(tmp_path)
        with patch.object(win, "_start_next") as start_next:
            win._on_files_selected([img])
            start_next.assert_not_called()
        # But the file IS in the queue.
        paths_in_queue = [getattr(item, "path", None) for item in win.queue._items]
        assert img in paths_in_queue, (
            f"expected {img} in queue; queue contents: {paths_in_queue}"
        )
    finally:
        win.close()


def test_start_button_still_triggers_run(tmp_path):
    """Verify the Start button (settings_panel.start_clicked) still calls
    _start_next. This pins that we didn't break the only working trigger."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        img = _make_image(tmp_path)
        win._on_files_selected([img])
        with patch.object(win, "_start_next") as start_next:
            win.settings_panel.start_clicked.emit()
            start_next.assert_called_once()
    finally:
        win.close()
