"""GenerateLocallyDialog shows the back-prop recommendation in the
preset description label and bumps the preset's effective
max_resolution (#131)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def big_card_probe(monkeypatch):
    """Monkey-patch probe_free_vram to return canned values matching an
    idle RTX PRO 6000 Blackwell — back-prop should suggest a value
    well above 720."""
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=True,
            free_mib=95 * 1024, total_mib=98 * 1024,
            name="NVIDIA RTX PRO 6000 Blackwell",
            driver_version="555.85",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )


@pytest.fixture
def probe_unavailable(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="nvidia-smi not on PATH",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )


def test_label_shows_recommendation_on_supported_gpu(qapp, big_card_probe):
    dlg = GenerateLocallyDialog(parent=None)
    desc_text = dlg.preset_desc.text()
    assert "Recommended" in desc_text
    # Big-card scenario: back-prop should raise above 720.
    # The label includes the free-GiB context.
    assert "GiB free" in desc_text
    dlg.deleteLater()


def test_label_floor_message_when_probe_unavailable(qapp, probe_unavailable):
    dlg = GenerateLocallyDialog(parent=None)
    desc_text = dlg.preset_desc.text()
    assert "Recommended" in desc_text
    # Probe-unavailable scenario: should show the floor message.
    assert "safety floor" in desc_text or "probe unavailable" in desc_text.lower()
    dlg.deleteLater()


def test_preset_data_carries_effective_max_resolution(qapp, big_card_probe):
    """The preset combo's currentData() should reflect the BUMPED
    max_resolution so build_run_config gets the back-prop value."""
    dlg = GenerateLocallyDialog(parent=None)
    preset = dlg.preset_combo.currentData()
    assert preset is not None
    # Big card → max_resolution should be at least the preset's baked
    # value (back-prop never lowers).
    baked = preset.get("baked_max_resolution", preset["max_resolution"])
    assert preset["max_resolution"] >= baked
    dlg.deleteLater()


def test_preset_change_recomputes_recommendation(qapp, big_card_probe):
    """Changing the preset (different K) triggers a new probe + bump."""
    dlg = GenerateLocallyDialog(parent=None)
    # Lineart (K=4096) vs Hi-Res (K=12288) at the same probe → different
    # recommendations (lower K → more headroom per candidate → higher max_res).
    dlg.preset_combo.setCurrentIndex(0)   # Lineart
    lineart_max = dlg.preset_combo.currentData()["max_resolution"]
    dlg.preset_combo.setCurrentIndex(3)   # Hi-Res
    hires_max = dlg.preset_combo.currentData()["max_resolution"]
    # Both should be ≥ their baked values; lineart should generally be
    # higher because K is smaller.
    assert lineart_max >= 480   # Lineart baked
    assert hires_max >= 1000    # Hi-Res baked
    dlg.deleteLater()
