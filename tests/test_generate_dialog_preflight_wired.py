"""GenerateLocallyDialog._on_generate_clicked must call gpu_run_preflight
and NOT accept (close with QDialog.Accepted) if proceed=False. If proceed
is True and the helper lowered max_resolution, .values() must reflect
the lowered value.

Correction Task 5 (VRAM honesty plan): the dialog used to bypass the
VRAM preflight entirely and spawn the worker directly. This test pins
the wire-up.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch  # noqa: E402

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(tmp_path):
    """Build a dialog with a real source path so the generate-button
    gate doesn't short-circuit _on_generate_clicked."""
    from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(
        parent=None,
        initial_source_path=img,
        gpu_budget_gib=24.0,
    )
    return dlg


def test_generate_dialog_blocks_accept_on_preflight_block(qapp, tmp_path):
    dlg = _make_dialog(tmp_path)
    try:
        with patch(
            "forza_abyss_painter.gui.generate_dialog.gpu_run_preflight",
            return_value=(False, dlg.values()),
        ) as preflight:
            dlg._on_generate_clicked()
            preflight.assert_called_once()
            assert dlg.result() != QDialog.DialogCode.Accepted
            # Worker must NOT have been constructed.
            assert not hasattr(dlg, "_worker") or dlg._worker is None
    finally:
        dlg.deleteLater()


def test_generate_dialog_values_reflects_lowered_max_res_after_proceed(
    qapp, tmp_path,
):
    """When preflight returns proceed=True with a lowered max_resolution,
    the dialog's .values() exposes the effective (lowered) value to the
    caller (so build_run_config sees the right number).

    We stop short of letting _on_generate_clicked actually spawn the worker
    by patching embedded_python_exe to point at a non-existent path — the
    real-path early-return surfaces a modal and the dialog stays open,
    which is fine; we only care that .values() reflects the lowered res.
    """
    dlg = _make_dialog(tmp_path)
    try:
        baseline = dict(dlg.values())
        lowered = dict(baseline)
        lowered["max_resolution"] = 480

        from pathlib import Path
        # Force the embedded-python early-return so the worker never spawns.
        with patch(
            "forza_abyss_painter.gui.generate_dialog.gpu_run_preflight",
            return_value=(True, lowered),
        ), patch(
            "forza_abyss_painter.gui.generate_dialog.embedded_python_exe",
            return_value=Path("/nonexistent/python.exe"),
        ), patch(
            "forza_abyss_painter.gui.generate_dialog.QMessageBox.critical",
        ):
            dlg._on_generate_clicked()
            # The effective max_resolution must now flow through .values().
            assert dlg.values()["max_resolution"] == 480
    finally:
        dlg.deleteLater()
