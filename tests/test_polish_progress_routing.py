"""Polish worker's progress + checkpoint signals must update the
middle PreviewPanel, matching CPU + GPU-fresh + Resume routing
(Chunked-K Task 4).

Before this wiring, Polish updated only the bottom QMainWindow
status bar via _on_polish_progress; the middle panel sat idle.
After: the worker's .progress and .checkpoint signals ALSO connect
to self.preview.on_progress, and _on_polish_started/_done/_error
update self.preview.status_label + self.preview.progress.
"""
import os
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch, MagicMock
from pathlib import Path
from PySide6.QtWidgets import QApplication


def test_polish_worker_progress_wires_to_preview_panel(tmp_path):
    """When _on_polish_requested spawns its worker, the worker's
    .progress and .checkpoint signals connect to self.preview.on_progress
    (in ADDITION to the existing _on_polish_progress hookups)."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    snap_json = tmp_path / "doc.json"
    snap_json.write_text(
        '{"format":"fd6.shapes","version":1,"image_size":[720,480],"shapes":[]}'
    )

    win = MainWindow()
    try:
        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(True, {"peak_gib": 2.0, "chunks_per_shape": 1, "free_gib": 24.0}),
        ), patch(
            "forza_abyss_painter.gui.gpu_gen_worker.GpuGenWorker"
        ) as Worker, patch(
            "forza_abyss_painter.gui.main_window.QThread"
        ) as Thread, patch(
            "forza_abyss_painter.gui.main_window.MainWindow._resolve_source_for_loaded_json",
            return_value=Path(str(tmp_path / "fake_source.png")),
        ), patch(
            "forza_abyss_painter.gui.runtime_install_dialog.prompt_install_or_use_existing",
            return_value=True,
        ), patch(
            "forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
        ) as ExePath, patch(
            "forza_abyss_painter.gui.polish_dialog.PolishDialog"
        ) as DlgCls:
            from PySide6.QtWidgets import QDialog
            ExePath.return_value = MagicMock(spec=Path)
            ExePath.return_value.exists.return_value = True
            dlg = DlgCls.return_value
            dlg.exec.return_value = QDialog.DialogCode.Accepted
            dlg.values.return_value = {
                "output_path": tmp_path / "out.json",
                "steps": 60,
                "lock_alpha": True,
            }
            Thread.return_value = MagicMock()

            win._on_polish_requested(snap_json)

            worker_instance = Worker.return_value
            progress_connects = [
                c.args[0] for c in worker_instance.progress.connect.call_args_list
            ]
            checkpoint_connects = [
                c.args[0] for c in worker_instance.checkpoint.connect.call_args_list
            ]

            assert win.preview.on_progress in progress_connects, (
                f"polish worker.progress missing preview.on_progress; "
                f"got {progress_connects}"
            )
            assert win.preview.on_progress in checkpoint_connects, (
                f"polish worker.checkpoint missing preview.on_progress; "
                f"got {checkpoint_connects}"
            )
            # And the existing bottom-bar hookup must NOT have been
            # removed by accident — middle panel + bottom must BOTH be
            # wired so the user sees status in both places.
            assert win._on_polish_progress in progress_connects, (
                "polish worker.progress lost its _on_polish_progress "
                "hookup (bottom status bar)"
            )
            assert win._on_polish_progress in checkpoint_connects, (
                "polish worker.checkpoint lost its _on_polish_progress "
                "hookup (bottom status bar)"
            )
    finally:
        win.close()


def test_polish_started_updates_preview_status_label():
    """_on_polish_started must update self.preview.status_label so the
    middle panel reflects that polish is now running."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        # Seed a non-polish state first so we can verify the slot moved
        # it to a polish-labeled state.
        win.preview.status_label.setText("Idle.")
        win.preview.progress.setValue(50)

        win._on_polish_started({"total_steps": 60, "image_size": [720, 480]})

        text = win.preview.status_label.text().lower()
        assert "polish" in text, (
            f"preview.status_label didn't reflect polish start: {text!r}"
        )
        assert win.preview.progress.value() == 0, (
            "preview.progress should reset to 0 at polish start; "
            f"got {win.preview.progress.value()}"
        )
    finally:
        win.close()


def test_polish_done_sets_preview_progress_to_complete(tmp_path):
    """_on_polish_done must set self.preview.progress to 100 and update
    self.preview.status_label, so the middle panel ends on a 'done'
    state instead of being abandoned mid-run."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        out = tmp_path / "polished.json"
        out.write_text(
            '{"format":"fd6.shapes","version":1,"image_size":[720,480],"shapes":[]}'
        )
        with patch.object(win, "_on_json_loaded_for_preview"):
            win._on_polish_done(str(out), 100)
        assert win.preview.progress.value() == 100, (
            f"preview.progress not at 100 after polish done: "
            f"{win.preview.progress.value()}"
        )
        text = win.preview.status_label.text().lower()
        assert "polish" in text and "done" in text, (
            f"preview.status_label didn't reflect polish done: {text!r}"
        )
    finally:
        win.close()


def test_polish_error_updates_preview_status_label():
    """_on_polish_error must update self.preview.status_label so the
    middle panel surfaces the failure state, not a stale in-progress
    label."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.preview.status_label.setText("Polishing — running joint_polish optimizer…")
        win.preview.progress.setValue(42)
        with patch("PySide6.QtWidgets.QMessageBox.critical"):
            win._on_polish_error("joint_polish", "boom")
        text = win.preview.status_label.text().lower()
        assert "polish" in text and ("fail" in text or "error" in text), (
            f"preview.status_label didn't reflect polish error: {text!r}"
        )
        assert win.preview.progress.value() == 0, (
            "preview.progress should reset to 0 on polish error; "
            f"got {win.preview.progress.value()}"
        )
    finally:
        win.close()
