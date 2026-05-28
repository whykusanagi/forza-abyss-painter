"""GenerateLocallyDialog preset combo + description show LIVE
estimate_full_pipeline_gib, not the hardcoded est_peak_vram_gib
marketing number per preset."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.generate_dialog import (
    GenerateLocallyDialog, LOCAL_PRESETS,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_preset_combo_label_shows_live_estimate(qapp, monkeypatch):
    """Preset combo label format includes a peak-VRAM number derived
    from estimate_full_pipeline_gib(K, max_res), NOT the static
    est_peak_vram_gib marketing value (12.0 etc.)."""
    # Patch probe so the test is reproducible.
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod
    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="test", probed_at=0.0,
        )
    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )

    dlg = GenerateLocallyDialog(parent=None)
    # The Hi-Res preset's static est_peak_vram_gib is 12.0 — but the
    # live full-pipeline estimate is ~165 GiB. Confirm the combo
    # label is NOT 12.0.
    hi_res_idx = next(
        i for i, p in enumerate(LOCAL_PRESETS) if "Hi-Res" in p["label"]
    )
    dlg.preset_combo.setCurrentIndex(hi_res_idx)
    label = dlg.preset_combo.itemText(hi_res_idx)
    # Old format: "Hi-Res — 3000 shapes  (~12.0 GiB peak)"
    # New format: "Hi-Res — 3000 shapes  (~165 GiB peak full pipeline)" or similar
    assert "12.0 GiB" not in label, (
        f"Preset label still shows static 12.0 GiB: {label!r}"
    )
    # Confirm a non-trivial peak number is present (Hi-Res full-pipeline ≥ 100 GiB).
    import re
    match = re.search(r"~(\d+(?:\.\d+)?)\s*GiB", label)
    assert match is not None, f"No peak number in label: {label!r}"
    peak_in_label = float(match.group(1))
    assert peak_in_label >= 100.0, (
        f"Live label peak {peak_in_label} GiB is unrealistically low for Hi-Res "
        f"(expected ~165 GiB). Label: {label!r}"
    )
    dlg.deleteLater()


def test_description_label_uses_live_estimate(qapp, monkeypatch):
    """The preset description label (preset_desc) also shows live
    estimate, not the static field."""
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod
    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="test", probed_at=0.0,
        )
    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )

    dlg = GenerateLocallyDialog(parent=None)
    hi_res_idx = next(
        i for i, p in enumerate(LOCAL_PRESETS) if "Hi-Res" in p["label"]
    )
    dlg.preset_combo.setCurrentIndex(hi_res_idx)
    desc = dlg.preset_desc.text()
    # The description should NOT show "12.0 GiB" — the old static value.
    assert "12.0 GiB" not in desc, (
        f"Description still shows static 12.0 GiB: {desc!r}"
    )
    dlg.deleteLater()
