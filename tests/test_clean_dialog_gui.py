"""GUI-side tests for the fap-clean Tools menu integration (#60).

Real PySide6 — same pattern as test_gpu_bundle_gui.py. Mocks only the
file-picker boundary (QFileDialog) so the dialog construction, summary
rendering, and save flow all exercise real Qt. The cleanup math itself
is covered by tests/test_cli_clean.py — these tests pin the GUI wiring
that calls into it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


# Defensive shim — modal QMessageBox.critical() can block forever in some
# offscreen Qt builds. Tests that EXPECT an error path monkeypatch this
# fixture-locally to capture the call; tests that don't expect one are
# protected by this fixture so a fixture drift surfaces as a real assertion
# failure rather than a silent hang.
@pytest.fixture(autouse=True)
def _no_block_messageboxes(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **kw: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **kw: QMessageBox.Ok))
    yield


# ----------------------------------------------- fixtures: docs that cleanup-want


def _doc_with_padding_white_and_dead_shape(tmp_path: Path) -> Path:
    """Write a small JSON containing one obvious padding-white shape
    (inside the 8% pad margin, color = 255,255,255) and one dead shape
    (fully occluded by a larger opaque shape covering it). Saved as
    JSON the dialog flow will load."""
    # Image 200x200, padding margin = 16 px (8% of 200).
    # Shape 0: padding-white at (5, 5) — inside margin, white → dropped
    # Shape 1: small interior shape that gets buried under shape 2
    # Shape 2: large opaque shape covering shape 1
    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "sticker_mode": False,
        "image_size": [200, 200],
        "shape_count": 3,
        "shapes": [
            {"type": "rotated_ellipse", "x": 5, "y": 5,
             "rx": 4, "ry": 4, "angle": 0,
             "color": [255, 255, 255, 255], "score": 0.0},
            {"type": "rotated_ellipse", "x": 100, "y": 100,
             "rx": 5, "ry": 5, "angle": 0,
             "color": [50, 50, 50, 255], "score": 0.0},
            {"type": "rotated_ellipse", "x": 100, "y": 100,
             "rx": 60, "ry": 60, "angle": 0,
             "color": [200, 200, 200, 255], "score": 0.0},
        ],
    }
    src = tmp_path / "input.json"
    src.write_text(json.dumps(doc), encoding="utf-8")
    return src


# --------------------------------------------------- helper: format_report


def test_format_report_includes_all_breakdown_counts():
    """The summary text must surface every count the user needs to
    decide whether to save: input, output, total dropped, and the
    three breakdown categories. If a refactor drops one of these
    silently the user loses visibility into what changed."""
    from forza_abyss_painter.gui.clean_dialog import _format_report
    report = {
        "input_count": 100, "output_count": 70, "dropped_total": 30,
        "dropped_padding_whites_only": 12,
        "dropped_dead_shapes_only": 15,
        "dropped_both_conditions": 3,
    }
    text = _format_report(report, "test.json")
    assert "100" in text and "70" in text and "30" in text
    assert "12" in text and "15" in text and "3" in text
    assert "test.json" in text


# --------------------------------------------------- dialog construction


def test_clean_dialog_constructs_with_default_output_suffix(tmp_path):
    """Default output path uses `_cleaned` suffix in source dir —
    matches the fap-clean CLI's default naming. If we ever switch to
    overwrite-in-place by default users will lose source files."""
    from forza_abyss_painter.gui.clean_dialog import CleanJsonDialog
    src = tmp_path / "logo.json"
    src.write_text("{}", encoding="utf-8")
    report = {
        "input_count": 10, "output_count": 8, "dropped_total": 2,
        "dropped_padding_whites_only": 1,
        "dropped_dead_shapes_only": 1, "dropped_both_conditions": 0,
    }
    d = CleanJsonDialog(None, src, {"shapes": [], "image_size": [10, 10]}, report)
    expected = tmp_path / "logo_cleaned.json"
    assert d.output_field.text() == str(expected), (
        f"output field {d.output_field.text()!r} != expected {expected!r} "
        f"— default suffix drifted from CLI behavior"
    )


def test_clean_dialog_summary_panel_shows_real_counts(tmp_path):
    """Summary panel must render the actual report dict, not a
    placeholder. A user staring at "X dropped" with X=0 might save
    needlessly; the panel is the only signal they have pre-save."""
    from forza_abyss_painter.gui.clean_dialog import CleanJsonDialog
    src = tmp_path / "x.json"
    src.write_text("{}", encoding="utf-8")
    report = {
        "input_count": 999, "output_count": 555, "dropped_total": 444,
        "dropped_padding_whites_only": 200,
        "dropped_dead_shapes_only": 234, "dropped_both_conditions": 10,
    }
    d = CleanJsonDialog(None, src, {"shapes": [], "image_size": [10, 10]}, report)
    text = d.summary.text()
    assert "999" in text
    assert "555" in text
    assert "444" in text
    assert "200" in text
    assert "234" in text


# --------------------------------------------------- save flow


def test_clean_dialog_save_writes_cleaned_doc_to_chosen_path(tmp_path):
    """Clicking Save persists the cleaned doc to disk at the path in
    the output field. This is the dialog's primary side effect — if
    it silently no-ops, users lose their cleanup work."""
    from forza_abyss_painter.gui.clean_dialog import CleanJsonDialog
    from forza_abyss_painter.io.exporter import load_json
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    cleaned = {
        "format": "fd6.shapes",
        "version": 1,
        "sticker_mode": False,
        "image_size": [50, 50],
        "shape_count": 1,
        "shapes": [{"type": "rotated_ellipse", "x": 10, "y": 10,
                    "rx": 3, "ry": 3, "angle": 0,
                    "color": [100, 100, 100, 255], "score": 0.0}],
    }
    report = {
        "input_count": 3, "output_count": 1, "dropped_total": 2,
        "dropped_padding_whites_only": 1,
        "dropped_dead_shapes_only": 1, "dropped_both_conditions": 0,
    }
    d = CleanJsonDialog(None, src, cleaned, report)
    out_path = tmp_path / "custom_out.json"
    d.output_field.setText(str(out_path))
    d._on_save()
    assert out_path.exists(), "save handler didn't write the file"
    assert d.output_path == out_path, (
        "dialog didn't expose output_path on accept — caller can't chain"
    )
    # Round-trip verify the written content matches what we passed in.
    loaded = load_json(out_path)
    assert len(loaded.shapes) == 1


def test_clean_dialog_save_with_empty_path_warns_and_stays_open(tmp_path, monkeypatch):
    """Empty output field → block save, surface a warning, leave the
    dialog open. Avoids silently writing to '' (which would either
    crash or land in cwd, both bad)."""
    from forza_abyss_painter.gui.clean_dialog import CleanJsonDialog
    from PySide6.QtWidgets import QMessageBox
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    report = {
        "input_count": 1, "output_count": 1, "dropped_total": 0,
        "dropped_padding_whites_only": 0,
        "dropped_dead_shapes_only": 0, "dropped_both_conditions": 0,
    }
    d = CleanJsonDialog(None, src, {"shapes": [], "image_size": [10, 10]}, report)
    d.output_field.setText("")
    warn_calls = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **kw: warn_calls.append(a) or QMessageBox.Ok)
    d._on_save()
    assert warn_calls, "empty path didn't trigger a warning"
    assert d.output_path is None, "output_path leaked despite empty input"


# --------------------------------------------------- entry point flow


def test_open_clean_json_dialog_cancels_on_picker_cancel(monkeypatch):
    """If the file picker returns empty (user cancelled), the entry
    point returns None without spinning up the summary dialog."""
    from forza_abyss_painter.gui import clean_dialog as cd
    monkeypatch.setattr(
        cd.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **kw: ("", "")),
    )
    constructed = [0]
    real = cd.CleanJsonDialog
    class _Spy(real):
        def __init__(self, *a, **kw):
            constructed[0] += 1
            super().__init__(*a, **kw)
    monkeypatch.setattr(cd, "CleanJsonDialog", _Spy)
    assert cd.open_clean_json_dialog(None) is None
    assert constructed[0] == 0


def test_open_clean_json_dialog_full_flow_returns_saved_path(monkeypatch, tmp_path):
    """End-to-end: picker selects a real JSON, clean_doc runs, dialog
    is constructed with the report, save flow writes the output. The
    return value is the saved path — caller chains it into the
    preview-load handler."""
    src = _doc_with_padding_white_and_dead_shape(tmp_path)
    out_path = tmp_path / "input_cleaned.json"

    from forza_abyss_painter.gui import clean_dialog as cd
    monkeypatch.setattr(
        cd.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(src), "JSON (*.json)")),
    )
    # Auto-accept the dialog: replace exec with a function that calls
    # _on_save and returns Accepted. The real save handler does the
    # actual write — we're just skipping the user's mouse click.
    real_exec = cd.CleanJsonDialog.exec
    def _auto_save(self):
        self._on_save()
        return QDialog.Accepted
    monkeypatch.setattr(cd.CleanJsonDialog, "exec", _auto_save)

    result = cd.open_clean_json_dialog(None)
    assert result == out_path, (
        f"entry point returned {result!r}, expected {out_path!r}"
    )
    assert out_path.exists()


# --------------------------------------------------- main_window wiring


def test_main_window_tools_menu_has_clean_action():
    """Wiring smoke: main_window's Tools menu contains the clean
    action. Grep-style like the generate-action test to keep it
    independent of MainWindow's heavy constructor."""
    import re
    src = (Path(__file__).resolve().parent.parent /
           "forza_abyss_painter" / "gui" / "main_window.py"
           ).read_text(encoding="utf-8")
    assert "Clean current JSON" in src, "menu label missing"
    assert "_on_clean_json" in src, "handler not referenced"
    assert re.search(r"clean_act\.triggered\.connect\(self\._on_clean_json\)", src), (
        "menu action not wired to handler"
    )
