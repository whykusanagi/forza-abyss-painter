"""Source-image resolution for #85/#86: try <json_dir>/<doc.source_image>
first; return None if missing so the caller can open a file picker."""
from __future__ import annotations

from pathlib import Path

from forza_abyss_painter.gui.main_window import _resolve_source_image_path


def test_sibling_exists_returns_sibling_path(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    image = tmp_path / "nikke.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = _resolve_source_image_path(json_path, "nikke.png")
    assert resolved == image


def test_sibling_missing_returns_none(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")

    resolved = _resolve_source_image_path(json_path, "nikke.png")
    assert resolved is None


def test_empty_source_image_returns_none(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")

    resolved = _resolve_source_image_path(json_path, "")
    assert resolved is None


def test_source_image_with_pathlike_chars_uses_basename(tmp_path):
    # Defensive: source_image is supposed to be a filename only, but if
    # a malformed JSON has "subdir/nikke.png", we still resolve to the
    # sibling rather than executing the subpath.
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    image = tmp_path / "nikke.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = _resolve_source_image_path(json_path, "subdir/nikke.png")
    assert resolved == image
