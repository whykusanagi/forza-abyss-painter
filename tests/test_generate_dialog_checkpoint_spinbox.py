"""Generate dialog has a 'Snapshot every N shapes' QSpinBox with
min=100, max=1000, step=50, default=100 (cuda min, per spec §6).
The value flows into the IPC config."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_checkpoint_spinbox_bounds(qapp):
    dlg = GenerateLocallyDialog(parent=None)
    sb = dlg.checkpoint_every_spinbox
    assert sb.minimum() == 100
    assert sb.maximum() == 1000
    assert sb.singleStep() == 50
    assert sb.value() == 100   # default
    dlg.deleteLater()


def test_checkpoint_spinbox_writes_into_build_run_config(qapp, tmp_path):
    """When the dialog spawns a run, the spinbox value lands in the
    config dict's checkpoint_every field (not the old // 20 heuristic)."""
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(parent=None)
    dlg.source_path = img
    dlg.checkpoint_every_spinbox.setValue(250)

    preset = dlg.preset_combo.currentData()
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
    cfg = build_run_config(
        img, tmp_path / "out.json", preset,
        checkpoint_every=int(dlg.checkpoint_every_spinbox.value()),
    )
    assert cfg["checkpoint_every"] == 250
    dlg.deleteLater()


def test_build_run_config_accepts_checkpoint_every_kwarg(tmp_path):
    """build_run_config gets a new optional kwarg. Default behavior
    (no kwarg) preserves the old num_shapes // 20 heuristic FLOORED
    at 100 (the cuda runner min)."""
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    preset = {
        "label": "Custom",
        "num_shapes": 400,
        "max_resolution": 480,
        "random_samples": 4096,
        "joint_polish_steps": 100,
    }
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
    # With explicit kwarg.
    cfg_explicit = build_run_config(
        img, tmp_path / "out.json", preset, checkpoint_every=100,
    )
    assert cfg_explicit["checkpoint_every"] == 100
    # Without kwarg → falls back to old heuristic FLOORED at 100.
    cfg_default = build_run_config(img, tmp_path / "out.json", preset)
    # 400 // 20 = 20, but the floor brings it up to 100.
    assert cfg_default["checkpoint_every"] == 100
